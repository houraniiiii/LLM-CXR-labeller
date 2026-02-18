from __future__ import annotations

import json
import logging
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Hashable, Iterable, Mapping, Optional, Tuple

try:
    from huggingface_hub import HfApi  # type: ignore
except ImportError as exc:  # pragma: no cover - huggingface_hub is a documented dependency.
    HfApi = None  # type: ignore[assignment]
    _HF_IMPORT_ERROR = exc
else:
    try:
        from huggingface_hub import HfHubError  # type: ignore[attr-defined]
    except ImportError:
        try:
            from huggingface_hub.utils import HfHubError  # type: ignore
        except ImportError:  # pragma: no cover - fallback for future hubs.
            class HfHubError(Exception):
                """Fallback error type when huggingface_hub does not expose HfHubError."""

                pass
    _HF_IMPORT_ERROR = None

try:
    from vllm import LLM  # type: ignore
except ImportError as exc:  # pragma: no cover - vLLM must be installed for inference.
    LLM = None  # type: ignore[assignment]
    _VLLM_IMPORT_ERROR = exc
else:
    _VLLM_IMPORT_ERROR = None

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional for static checks.
    torch = None


LOGGER = logging.getLogger(__name__)

MODELS_PATH = Path("models.jsonl")

QUANT_MODE_ALIASES: Dict[str, str] = {
    "fp16": "fp16",
    "float16": "fp16",
    "half": "fp16",
    "int4": "int4",
    "4bit": "int4",
    "bit4": "int4",
}

SUPPORTED_K_SHOTS: Tuple[int, ...] = (0, 5, 10)
SUPPORTED_QUANT_MODES = {"fp16", "int4"}

ALLOWED_VLLM_OVERRIDE_KEYS = {
    "dtype",
    "tensor_parallel_size",
    "gpu_memory_utilization",
    "max_model_len",
    "max_num_batched_tokens",
    "enforce_eager",
    "swap_space_gb",
    "kv_cache_dtype",
    "download_dir",
    "rope_scaling",
}

CacheKey = Tuple[str, str, Tuple[Tuple[str, Hashable], ...]]


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model_name: str
    tokenizer_id: str
    batch_sizes: Dict[str, Dict[int, int]]
    revision: Optional[str]
    trust_remote_code: bool
    reasoning_parser: Optional[str] = None

    def default_batch_size(self, quant_mode: str, k_shot: int) -> int:
        normalized_quant = normalize_quant_mode(quant_mode)
        try:
            k_shot_map = self.batch_sizes[normalized_quant]
        except KeyError as exc:
            raise KeyError(
                f"No batch size defaults defined for quant_mode '{normalized_quant}' "
                f"in model '{self.model_name}'."
            ) from exc
        try:
            return k_shot_map[k_shot]
        except KeyError as exc:
            available = ", ".join(str(value) for value in sorted(k_shot_map))
            raise ValueError(
                f"No batch size configured for k_shot={k_shot} under quant_mode '{normalized_quant}' "
                f"in model '{self.model_name}'. Available k_shot values: {available}."
            ) from exc


@dataclass(frozen=True)
class LoadedEngine:
    engine: Any
    tokenizer: Any
    model_spec: ModelSpec
    quant_mode: str
    repo_id: str
    revision: Optional[str]
    quant_method: str
    reasoning_parser: Optional[str]
    engine_options: Dict[str, Any]
    quant_options: Dict[str, Any]
    cache_key: CacheKey


def normalize_quant_mode(value: str) -> str:
    try:
        return QUANT_MODE_ALIASES[value.strip().lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported quant_mode '{value}'. Expected one of {sorted(SUPPORTED_QUANT_MODES)}."
        ) from exc


class VLLMEngineRegistry:
    """
    Manage vLLM engine instantiation and cache by (model, quant_mode, overrides).

    Quantization policy:
    - fp16: base repo, no quantization argument
    - int4: base repo, quantization='bitsandbytes'
    """

    def __init__(
        self,
        models_path: Path = MODELS_PATH,
        *,
        hf_api: Optional[Any] = None,
    ) -> None:
        if _VLLM_IMPORT_ERROR is not None:
            raise ImportError(
                "vLLM>=0.5 with CUDA 12.9 build is required. Install via "
                '"pip install \'vllm-cu129>=0.5\'" before creating the registry.'
            ) from _VLLM_IMPORT_ERROR
        if _HF_IMPORT_ERROR is not None:
            raise ImportError(
                "huggingface_hub is required to validate model availability."
            ) from _HF_IMPORT_ERROR

        self.models_path = Path(models_path)
        self._specs = self._load_specs()
        self._engine_cache: Dict[CacheKey, LoadedEngine] = {}
        self._repo_probe_cache: Dict[Tuple[str, Optional[str]], bool] = {}
        self._hf_api = hf_api or HfApi()

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #
    def list_model_names(self) -> Iterable[str]:
        return sorted(self._specs)

    def list_specs(self, provider: Optional[str] = None) -> Iterable[ModelSpec]:
        if provider is None:
            return [self._specs[name] for name in self.list_model_names()]
        provider_norm = provider.strip().lower()
        return [
            spec
            for spec in self._specs.values()
            if spec.provider.lower() == provider_norm
        ]

    def get(self, model_name: str) -> ModelSpec:
        try:
            return self._specs[model_name]
        except KeyError as exc:
            raise KeyError(f"Model '{model_name}' not found in {self.models_path}.") from exc

    def default_batch_size(self, model_name: str, quant_mode: str, k_shot: int) -> int:
        return self.get(model_name).default_batch_size(quant_mode, k_shot)

    def evict(self, cache_key: CacheKey) -> None:
        """Remove a cached engine (typically before re-instantiating with new overrides)."""
        cached = self._engine_cache.pop(cache_key, None)
        if cached is not None:
            self._shutdown_engine(cached)
            self._release_cuda_cache()

    def evict_model(self, model_name: str) -> int:
        """Evict all cached engines for a single model. Returns number of removed engines."""
        keys_to_evict = [key for key in self._engine_cache if key[0] == model_name]
        for key in keys_to_evict:
            cached = self._engine_cache.pop(key, None)
            if cached is not None:
                self._shutdown_engine(cached)
        if keys_to_evict:
            self._release_cuda_cache()
        return len(keys_to_evict)

    def evict_all(self) -> int:
        """Evict every cached engine. Returns number of removed engines."""
        keys_to_evict = list(self._engine_cache)
        for key in keys_to_evict:
            cached = self._engine_cache.pop(key, None)
            if cached is not None:
                self._shutdown_engine(cached)
        if keys_to_evict:
            self._release_cuda_cache()
        return len(keys_to_evict)

    def load_engine(
        self,
        model_name: str,
        quant_mode: str,
        *,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> LoadedEngine:
        spec = self.get(model_name)
        normalized_quant = normalize_quant_mode(quant_mode)
        merged_overrides = dict(overrides or {})

        cache_key = self._make_cache_key(model_name, normalized_quant, merged_overrides)
        cached = self._engine_cache.get(cache_key)
        if cached is not None:
            return cached

        repo_id, revision, quant_method, quant_options = self._resolve_repo(spec, normalized_quant)
        self._validate_repo_available(repo_id, revision)

        engine_kwargs = self._build_engine_kwargs(
            spec=spec,
            repo_id=repo_id,
            revision=revision,
            quant_mode=normalized_quant,
            quant_method=quant_method,
            quant_options=quant_options,
            overrides=merged_overrides,
        )

        LOGGER.debug(
            "Instantiating vLLM engine: model=%s quant=%s repo=%s revision=%s kwargs=%s",
            spec.model_name,
            normalized_quant,
            repo_id,
            revision,
            {key: value for key, value in engine_kwargs.items() if key not in {"tokenizer", "model"}},
        )

        engine = LLM(**engine_kwargs)
        tokenizer = engine.get_tokenizer()
        loaded = LoadedEngine(
            engine=engine,
            tokenizer=tokenizer,
            model_spec=spec,
            quant_mode=normalized_quant,
            repo_id=repo_id,
            revision=revision,
            quant_method=quant_method,
            reasoning_parser=spec.reasoning_parser,
            engine_options=merged_overrides,
            quant_options=quant_options,
            cache_key=cache_key,
        )
        self._engine_cache[cache_key] = loaded
        return loaded

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _make_cache_key(
        self,
        model_name: str,
        quant_mode: str,
        overrides: Mapping[str, Any],
    ) -> CacheKey:
        normalized_items: Tuple[Tuple[str, Hashable], ...] = tuple(
            sorted(
                (str(key), self._freeze_cache_value(value))
                for key, value in overrides.items()
            )
        )
        return (model_name, quant_mode, normalized_items)

    def _freeze_cache_value(self, value: Any) -> Hashable:
        if isinstance(value, Mapping):
            return tuple(
                sorted(
                    (str(key), self._freeze_cache_value(item_value))
                    for key, item_value in value.items()
                )
            )
        if isinstance(value, (list, tuple)):
            return tuple(self._freeze_cache_value(item) for item in value)
        if isinstance(value, set):
            frozen_items = [self._freeze_cache_value(item) for item in value]
            return tuple(sorted(frozen_items, key=repr))
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        raise TypeError(
            f"Unsupported override value type {type(value).__name__} for cache key construction."
        )

    @staticmethod
    def _shutdown_engine(loaded: LoadedEngine) -> None:
        try:
            loaded.engine.shutdown()  # type: ignore[attr-defined]
        except AttributeError:
            pass

    @staticmethod
    def _release_cuda_cache() -> None:
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _resolve_repo(
        self,
        spec: ModelSpec,
        quant_mode: str,
    ) -> Tuple[str, Optional[str], str, Dict[str, Any]]:
        if quant_mode == "fp16":
            return (spec.model_name, spec.revision, "fp16", {})
        if quant_mode == "int4":
            return (spec.model_name, spec.revision, "bitsandbytes", {})
        raise ValueError(f"Unsupported quant_mode '{quant_mode}'.")

    def _validate_repo_available(self, repo_id: str, revision: Optional[str]) -> None:
        cache_key = (repo_id, revision)
        if cache_key in self._repo_probe_cache:
            return
        try:
            self._hf_api.repo_info(repo_id, revision=revision or None)
        except HfHubError as exc:
            raise RuntimeError(
                f"Unable to access repo '{repo_id}' (revision={revision or 'default'}). "
                "Check that the repository exists and that you have permission to download it."
            ) from exc
        self._repo_probe_cache[cache_key] = True

    def _build_engine_kwargs(
        self,
        *,
        spec: ModelSpec,
        repo_id: str,
        revision: Optional[str],
        quant_mode: str,
        quant_method: str,
        quant_options: Dict[str, Any],
        overrides: Mapping[str, Any],
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": repo_id,
            "revision": revision,
            "tokenizer": spec.tokenizer_id,
            "tokenizer_revision": spec.revision,
            "trust_remote_code": spec.trust_remote_code,
        }

        if quant_mode == "int4":
            kwargs["quantization"] = quant_method

        for key, value in quant_options.items():
            if key not in ALLOWED_VLLM_OVERRIDE_KEYS:
                raise ValueError(
                    f"Unsupported quant option '{key}' provided for model '{spec.model_name}'."
                )
            kwargs[self._map_override_key(key)] = value

        for key, value in overrides.items():
            if key not in ALLOWED_VLLM_OVERRIDE_KEYS:
                raise ValueError(
                    f"Unsupported vLLM override '{key}'. Allowed keys: {sorted(ALLOWED_VLLM_OVERRIDE_KEYS)}."
                )
            if value is None:
                continue
            kwargs[self._map_override_key(key)] = value

        if spec.reasoning_parser:
            kwargs["reasoning_parser"] = spec.reasoning_parser

        return kwargs

    @staticmethod
    def _map_override_key(key: str) -> str:
        if key == "swap_space_gb":
            return "swap_space"
        return key

    def _load_specs(self) -> Dict[str, ModelSpec]:
        if not self.models_path.exists():
            raise FileNotFoundError(f"Model catalog not found at {self.models_path}")

        specs: Dict[str, ModelSpec] = {}
        with self.models_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {self.models_path} line {line_number}: {exc}") from exc
                spec = self._parse_spec(payload, line_number)
                specs[spec.model_name] = spec
        if not specs:
            raise ValueError(f"No model entries found in {self.models_path}.")
        return specs

    def _parse_spec(self, payload: Dict[str, Any], line_number: int) -> ModelSpec:
        required = {
            "provider",
            "model_name",
            "tokenizer_id",
            "batch_sizes",
            "trust_remote_code",
        }
        missing = required - payload.keys()
        if missing:
            raise ValueError(
                f"Missing fields {missing} in {self.models_path} line {line_number}."
            )

        batch_sizes = self._parse_batch_sizes(payload["batch_sizes"], line_number)

        revision = payload.get("revision")
        if revision is not None and not isinstance(revision, str):
            raise ValueError(f"revision must be a string or null in {self.models_path} line {line_number}.")

        reasoning_parser = payload.get("reasoning_parser")
        if reasoning_parser is not None and not isinstance(reasoning_parser, str):
            raise ValueError(
                f"reasoning_parser must be a string when provided in {self.models_path} line {line_number}."
            )

        return ModelSpec(
            provider=str(payload["provider"]),
            model_name=str(payload["model_name"]),
            tokenizer_id=str(payload["tokenizer_id"]),
            batch_sizes=batch_sizes,
            revision=revision,
            trust_remote_code=bool(payload["trust_remote_code"]),
            reasoning_parser=reasoning_parser,
        )

    def _parse_batch_sizes(self, raw: Any, line_number: int) -> Dict[str, Dict[int, int]]:
        if not isinstance(raw, dict):
            raise ValueError(
                f"batch_sizes must be a mapping in {self.models_path} line {line_number}."
            )

        parsed: Dict[str, Dict[int, int]] = {}
        for quant_key, mapping in raw.items():
            normalized_quant = normalize_quant_mode(str(quant_key))
            if normalized_quant not in SUPPORTED_QUANT_MODES:
                raise ValueError(
                    f"batch_sizes contains unsupported quant mode '{quant_key}' "
                    f"in {self.models_path} line {line_number}."
                )
            if normalized_quant in parsed:
                raise ValueError(
                    f"batch_sizes contains duplicate entries for quant mode '{normalized_quant}' "
                    f"in {self.models_path} line {line_number}."
                )
            if not isinstance(mapping, dict):
                raise ValueError(
                    f"batch_sizes[{quant_key!r}] must be a mapping of k-shot to batch size "
                    f"in {self.models_path} line {line_number}."
                )

            k_map: Dict[int, int] = {}
            for k_value, batch_size in mapping.items():
                try:
                    k_int = int(k_value)
                    batch_int = int(batch_size)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"batch_sizes[{quant_key!r}] contains non-integer key/value in "
                        f"{self.models_path} line {line_number}."
                    ) from exc
                if k_int not in SUPPORTED_K_SHOTS:
                    supported = ", ".join(str(v) for v in SUPPORTED_K_SHOTS)
                    raise ValueError(
                        f"batch_sizes[{quant_key!r}] includes unsupported k-shot {k_int}; "
                        f"supported values: {supported}."
                    )
                k_map[k_int] = batch_int

            missing = [value for value in SUPPORTED_K_SHOTS if value not in k_map]
            if missing:
                missing_str = ", ".join(str(value) for value in missing)
                raise ValueError(
                    f"batch_sizes[{quant_key!r}] missing k-shot entries {missing_str} "
                    f"in {self.models_path} line {line_number}."
                )

            parsed[normalized_quant] = k_map

        return parsed


__all__ = [
    "LoadedEngine",
    "ModelSpec",
    "SUPPORTED_K_SHOTS",
    "VLLMEngineRegistry",
    "normalize_quant_mode",
]
