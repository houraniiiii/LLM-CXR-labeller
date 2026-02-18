from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, TextIO, Tuple

import numpy as np

from .data_io import DatasetBundle, DatasetExample, LABEL_ORDER, load_dataset
from .few_shot_retrieval import (
    ALLOWED_K_SHOTS,
    DEFAULT_FEW_SHOT_POOL_PATH,
    RetrievalFewShotResolver,
)
from .engines import LoadedEngine, ModelSpec, VLLMEngineRegistry, normalize_quant_mode
from .prompts import (
    FewShotExample,
    PromptBuilder,
    PromptVariant,
    canonical_to_title,
)
from .utils import slugify

try:  # Optional torch import for reproducibility / memory logging.
    import torch
except ImportError:  # pragma: no cover - torch is optional for some static checks.
    torch = None

try:
    from vllm import SamplingParams  # type: ignore
    from vllm.sampling_params import StructuredOutputsParams  # type: ignore
except ImportError as exc:  # pragma: no cover - enforced at runtime.
    SamplingParams = None  # type: ignore[assignment]
    StructuredOutputsParams = None  # type: ignore[assignment]
    _VLLM_IMPORT_ERROR = exc
else:
    _VLLM_IMPORT_ERROR = None


OOM_KEYWORDS = ("out of memory", "cuda oom", "cuda out of memory", "oom during allocation")
FINISH_REASON_LENGTH = "length"
REASONING_ENABLED = "enabled"
REASONING_SUPPRESSED = "suppressed"
REASONING_ABSENT = "absent"


@dataclass
class GenerationConfig:
    max_new_tokens: int = 2000
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    repetition_penalty: float = 1.0
    reasoning_effort: Optional[str] = None


@dataclass
class RunConfig:
    model_name: str
    dataset: str
    input_context: str
    quant_mode: str
    base_prompt: Optional[str] = None
    k_shot: int = 0
    requested_k_shot: Optional[int] = None
    prompt_variant: Optional[str] = None
    custom_system_prompt: Optional[str] = None
    few_shot_preset: Optional[str] = None
    few_shot_archive_split: Optional[str] = None
    few_shot_archive_path: Optional[Path] = None
    few_shot_pool_path: Path = field(default_factory=lambda: DEFAULT_FEW_SHOT_POOL_PATH)
    strict_few_shot_lookup: bool = True
    batch_size: Optional[int] = None
    seed: int = 42
    limit: Optional[int] = None
    outdir: Path = field(default_factory=lambda: Path("results"))
    auto_reduce_on_oom: bool = True
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    vllm_options: Dict[str, Any] = field(default_factory=dict)
    use_structured_output: bool = True
    suppress_reasoning: bool = False


@dataclass
class ParseStats:
    success: int = 0
    repaired: int = 0
    failed: int = 0
    overflow_skipped: int = 0

    def total_attempted(self) -> int:
        return self.success + self.failed


@dataclass
class BatchResult:
    example_records: List[Dict[str, Any]]
    raw_outputs: List[str]
    reasoning_outputs: List[Optional[str]]
    input_token_counts: List[int]
    output_token_counts: List[int]
    report_token_counts: List[int]
    latency_ms: float
    tokens_per_second: float
    parse_flags: List[Tuple[bool, bool]]  # (valid, repaired)
    overflow_flags: List[bool]


@dataclass(frozen=True)
class RunPaths:
    results_root: Path
    preds_path: Path
    preds_tmp_path: Path
    log_path: Path


@dataclass
class RunMetrics:
    parse_stats: ParseStats = field(default_factory=ParseStats)
    auto_reduce_count: int = 0
    total_latency: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_report_only_tokens: int = 0


@dataclass(frozen=True)
class ReasoningState:
    status: str
    parser_name: Optional[str]
    use_structured_output: bool

    @property
    def enabled(self) -> bool:
        return self.status == REASONING_ENABLED


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def extract_first_json_object(text: str) -> Optional[str]:
    depth = 0
    start_idx = None
    for idx, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start_idx = idx
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start_idx is not None:
                    return text[start_idx : idx + 1]
    return None


LogFn = Callable[[str], None]


class InferenceRunner:
    def __init__(
        self,
        *,
        model_registry: Optional[VLLMEngineRegistry] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        prompt_variant: PromptVariant | str | None = None,
        custom_system_prompt: Optional[str] = None,
    ) -> None:
        if _VLLM_IMPORT_ERROR is not None or SamplingParams is None:  # pragma: no cover - defensive.
            raise ImportError(
                "vLLM>=0.5 (CUDA 12.9 build) is required for inference. "
                'Install via `pip install "vllm-cu129>=0.5"` before running.'
            ) from _VLLM_IMPORT_ERROR

        self.registry = model_registry or VLLMEngineRegistry()
        if prompt_builder is not None:
            self.prompt_builder = prompt_builder
        else:
            self.prompt_builder = self._build_prompt_builder(
                prompt_variant=prompt_variant,
                custom_system_prompt=custom_system_prompt,
            )

        self._default_prompt_builder = self.prompt_builder
        self.label_order = list(LABEL_ORDER)
        self._chat_template_mode_by_model: Dict[str, str] = {}
        self._canonical_to_title: Dict[str, str] = {}
        self._title_to_canonical: Dict[str, str] = {}
        for canonical in self.label_order:
            try:
                title_label = canonical_to_title(canonical)
            except KeyError:
                title_label = canonical.replace("_", " ").title()
            self._canonical_to_title[canonical] = title_label
            self._title_to_canonical[title_label] = canonical
        self.title_label_order = [self._canonical_to_title[label] for label in self.label_order]
        self._json_schema = self._build_json_schema()

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def run(self, config: RunConfig) -> Dict[str, Path]:
        set_global_seed(config.seed)

        dataset = load_dataset(
            config.dataset,
            config.input_context,
            limit=config.limit,
        )
        self._apply_prompt_overrides(config)
        self._coerce_k_shot(config)
        few_shot_resolver, few_shot_details = self._resolve_few_shot_retrieval(
            config=config,
            dataset_name=dataset.name,
        )
        few_shot_details["requested_k"] = config.k_shot

        model_spec, quant_mode, batch_size, loaded_engine, active_overrides = self._prepare_engine(config)
        self._configure_tokenizer(loaded_engine)
        run_paths = self._build_run_paths(
            config=config,
            dataset=dataset,
            model_spec=model_spec,
            quant_mode=quant_mode,
            batch_size=batch_size,
            few_shot_details=few_shot_details,
        )

        log_lines, log = self._make_logger()
        self._log_run_start(
            log=log,
            config=config,
            dataset=dataset,
            model_spec=model_spec,
            quant_mode=quant_mode,
            batch_size=batch_size,
            few_shot_details=few_shot_details,
            loaded_engine=loaded_engine,
            active_overrides=active_overrides,
        )

        with run_paths.preds_tmp_path.open("w", encoding="utf-8") as writer:
            metrics = self._run_batch_loop(
                writer=writer,
                dataset=dataset,
                model_spec=model_spec,
                quant_mode=quant_mode,
                config=config,
                loaded_engine=loaded_engine,
                batch_size=batch_size,
                few_shot_resolver=few_shot_resolver,
                few_shot_details=few_shot_details,
                active_overrides=active_overrides,
                log=log,
            )

        run_paths.preds_tmp_path.replace(run_paths.preds_path)
        self._log_run_summary(
            log=log,
            total_examples=len(dataset),
            preds_path=run_paths.preds_path,
            metrics=metrics,
        )
        self._score_run(
            log=log,
            preds_path=run_paths.preds_path,
            dataset_name=config.dataset,
            results_root=run_paths.results_root,
        )

        log_lines.append("")
        run_paths.log_path.write_text("\n".join(log_lines), encoding="utf-8")

        return {
            "predictions": run_paths.preds_path,
            "log": run_paths.log_path,
        }

    @staticmethod
    def _build_prompt_builder(
        *,
        prompt_variant: PromptVariant | str | None,
        custom_system_prompt: Optional[str],
    ) -> PromptBuilder:
        builder_kwargs: Dict[str, Any] = {}
        if custom_system_prompt is not None:
            builder_kwargs["system_prompt"] = custom_system_prompt
        elif prompt_variant is not None:
            builder_kwargs["prompt_variant"] = prompt_variant
        return PromptBuilder(**builder_kwargs)

    @staticmethod
    def _coerce_k_shot(config: RunConfig) -> None:
        requested_k = config.requested_k_shot if config.requested_k_shot is not None else config.k_shot
        try:
            config.k_shot = int(requested_k)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid k_shot value {requested_k!r}; expected one of {ALLOWED_K_SHOTS}.") from exc
        if config.k_shot not in ALLOWED_K_SHOTS:
            allowed = ", ".join(str(value) for value in ALLOWED_K_SHOTS)
            raise ValueError(f"k_shot={config.k_shot} is unsupported. Allowed values: {allowed}.")

    def _prepare_engine(
        self,
        config: RunConfig,
    ) -> Tuple[ModelSpec, str, int, LoadedEngine, Dict[str, Any]]:
        """Resolve model/quant settings and load the vLLM engine once for the run."""
        model_spec = self.registry.get(config.model_name)
        quant_mode = normalize_quant_mode(config.quant_mode)
        batch_size = config.batch_size or model_spec.default_batch_size(quant_mode, config.k_shot)
        requested_overrides = dict(config.vllm_options or {})
        loaded_engine = self.registry.load_engine(config.model_name, quant_mode, overrides=requested_overrides)
        active_overrides = dict(loaded_engine.engine_options)
        return model_spec, quant_mode, batch_size, loaded_engine, active_overrides

    @staticmethod
    def _configure_tokenizer(loaded_engine: LoadedEngine) -> None:
        tokenizer = loaded_engine.tokenizer
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "left"

    def _build_run_paths(
        self,
        *,
        config: RunConfig,
        dataset: DatasetBundle,
        model_spec: ModelSpec,
        quant_mode: str,
        batch_size: int,
        few_shot_details: Dict[str, Any],
    ) -> RunPaths:
        """Build deterministic output artifact paths for predictions and logs."""
        results_root = Path(config.outdir)
        preds_dir = results_root / "preds" / dataset.name
        logs_dir = results_root / "logs"
        preds_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        provider_slug = slugify(model_spec.provider)
        model_slug = slugify(model_spec.model_name.replace("/", "-"))
        ctx_slug = slugify(config.input_context)
        few_shot_slug = slugify(
            str(
                few_shot_details.get("mode")
                or ("retrieval" if config.k_shot > 0 else "zero-shot")
            )
        )
        prompt_slug = slugify(self.prompt_builder.prompt_variant_label or "clinical_standard")

        preds_filename = (
            f"{provider_slug}__{model_slug}__{quant_mode}"
            f"__k{config.k_shot}__ctx-{ctx_slug}"
            f"__fewshot-{few_shot_slug}__prompt-{prompt_slug}"
            f"__bs{batch_size}.jsonl"
        )
        log_filename = (
            f"{dataset.name}__{quant_mode}__{model_slug}"
            f"__k{config.k_shot}__ctx-{ctx_slug}"
            f"__fewshot-{few_shot_slug}__prompt-{prompt_slug}"
            f".log"
        )
        preds_path = preds_dir / preds_filename
        return RunPaths(
            results_root=results_root,
            preds_path=preds_path,
            preds_tmp_path=preds_path.with_suffix(".jsonl.tmp"),
            log_path=logs_dir / log_filename,
        )

    @staticmethod
    def _make_logger() -> Tuple[List[str], LogFn]:
        log_lines: List[str] = []

        def log(msg: str) -> None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            full = f"[{timestamp}] {msg}"
            print(full)
            log_lines.append(full)

        return log_lines, log

    def _log_run_start(
        self,
        *,
        log: LogFn,
        config: RunConfig,
        dataset: DatasetBundle,
        model_spec: ModelSpec,
        quant_mode: str,
        batch_size: int,
        few_shot_details: Dict[str, Any],
        loaded_engine: LoadedEngine,
        active_overrides: Dict[str, Any],
    ) -> None:
        log("Starting inference run.")
        log(f"Dataset={dataset.name} examples={len(dataset)}")
        log(f"Model={model_spec.model_name} (provider={model_spec.provider})")
        log(f"Quant mode={quant_mode} batch_size={batch_size}")
        log(
            f"k_shot={config.k_shot} (requested={few_shot_details['requested_k']}) "
            f"max_new_tokens={config.generation.max_new_tokens} "
            f"input_context={config.input_context}"
        )
        log(
            "Few-shot config: mode={mode} split={split} archive={archive} pool={pool} "
            "strict={strict} requested={requested}".format(
                mode=few_shot_details["mode"],
                split=few_shot_details["split"],
                archive=few_shot_details["archive"],
                pool=few_shot_details["pool"],
                strict=few_shot_details["strict"],
                requested=few_shot_details["requested_k"],
            )
        )
        log(
            f"Prompt variant={self.prompt_builder.prompt_variant_label} "
            f"few_shot_examples_target={config.k_shot}"
        )
        if config.custom_system_prompt is not None or self.prompt_builder.prompt_variant is None:
            log("Custom system prompt enabled for this run.")
        if config.base_prompt:
            log("Legacy base_prompt override ignored by prompt pipeline.")
        log(
            "Quant repo=%s revision=%s method=%s"
            % (
                loaded_engine.repo_id,
                loaded_engine.revision or "main",
                loaded_engine.quant_method,
            )
        )
        if active_overrides:
            log(f"vLLM options={json.dumps(active_overrides, sort_keys=True)}")
        if config.limit is not None:
            log(f"Limit applied: first {config.limit} examples.")

    def _run_batch_loop(
        self,
        *,
        writer: TextIO,
        dataset: DatasetBundle,
        model_spec: ModelSpec,
        quant_mode: str,
        config: RunConfig,
        loaded_engine: LoadedEngine,
        batch_size: int,
        few_shot_resolver: Optional[RetrievalFewShotResolver],
        few_shot_details: Dict[str, Any],
        active_overrides: Dict[str, Any],
        log: LogFn,
    ) -> RunMetrics:
        """Run batched generation and write prediction rows."""
        metrics = RunMetrics()
        total_examples = len(dataset)
        current_batch_size = batch_size
        index = 0
        last_reasoning_status: Optional[str] = None

        while index < total_examples:
            batch_examples = dataset.examples[index : index + current_batch_size]
            reasoning_state = self._resolve_reasoning_state(
                reasoning_parser_name=loaded_engine.reasoning_parser,
                suppress_reasoning=config.suppress_reasoning,
                use_structured_output=config.use_structured_output,
            )
            if reasoning_state.status != last_reasoning_status:
                status_message = self._format_reasoning_status_log(
                    reasoning_state=reasoning_state,
                    previous_status=last_reasoning_status,
                    config_use_structured_output=config.use_structured_output,
                )
                if status_message:
                    log(status_message)
                last_reasoning_status = reasoning_state.status

            batch_few_shots = self._resolve_batch_few_shots(
                batch_examples=batch_examples,
                dataset_name=dataset.name,
                resolver=few_shot_resolver,
                k_shot=config.k_shot,
            )

            try:
                batch_result = self._generate_batch(
                    batch_examples=batch_examples,
                    loaded_engine=loaded_engine,
                    gen_config=config.generation,
                    prompt_seed=config.seed,
                    use_structured_output=reasoning_state.use_structured_output,
                    few_shots_by_example=batch_few_shots,
                )
            except RuntimeError as err:
                if config.auto_reduce_on_oom and self._is_recoverable_oom(err):
                    metrics.auto_reduce_count += 1
                    loaded_engine, current_batch_size, active_overrides = self._handle_oom_recovery(
                        log=log,
                        previous_engine=loaded_engine,
                        current_batch_size=current_batch_size,
                        active_overrides=active_overrides,
                    )
                    continue
                raise

            self._write_batch_records(
                writer=writer,
                batch_examples=batch_examples,
                batch_few_shots=batch_few_shots,
                batch_result=batch_result,
                dataset_name=dataset.name,
                model_spec=model_spec,
                quant_mode=quant_mode,
                loaded_engine=loaded_engine,
                config=config,
                few_shot_details=few_shot_details,
                reasoning_enabled=reasoning_state.enabled,
                parse_stats=metrics.parse_stats,
            )
            self._accumulate_batch_metrics(metrics, batch_result)
            index += len(batch_examples)

        return metrics

    @staticmethod
    def _resolve_reasoning_state(
        *,
        reasoning_parser_name: Optional[str],
        suppress_reasoning: bool,
        use_structured_output: bool,
    ) -> ReasoningState:
        reasoning_available = bool(reasoning_parser_name)
        if reasoning_available and not suppress_reasoning:
            status = REASONING_ENABLED
        elif reasoning_available and suppress_reasoning:
            status = REASONING_SUPPRESSED
        else:
            status = REASONING_ABSENT

        return ReasoningState(
            status=status,
            parser_name=reasoning_parser_name,
            use_structured_output=False if status == REASONING_ENABLED else use_structured_output,
        )

    @staticmethod
    def _format_reasoning_status_log(
        *,
        reasoning_state: ReasoningState,
        previous_status: Optional[str],
        config_use_structured_output: bool,
    ) -> Optional[str]:
        """Return the same reasoning-status transition message emitted by prior behavior."""
        parser_name = reasoning_state.parser_name or "unknown"
        if reasoning_state.status == REASONING_ENABLED:
            if config_use_structured_output:
                return f"Reasoning parser '{parser_name}' active; structured outputs disabled."
            return f"Reasoning parser '{parser_name}' active; structured outputs already disabled by config."

        if reasoning_state.status == REASONING_SUPPRESSED:
            if config_use_structured_output:
                return (
                    f"Reasoning parser '{parser_name}' available but suppressed via --suppress-reasoning; "
                    "structured outputs remain enabled."
                )
            return (
                f"Reasoning parser '{parser_name}' available but suppressed via --suppress-reasoning; "
                "structured outputs stay disabled by config."
            )

        if previous_status == REASONING_ENABLED and config_use_structured_output:
            return "Reasoning parser unavailable; structured outputs restored."
        if previous_status in {REASONING_ENABLED, REASONING_SUPPRESSED}:
            return "Reasoning parser unavailable."
        return None

    def _resolve_batch_few_shots(
        self,
        *,
        batch_examples: Sequence[DatasetExample],
        dataset_name: str,
        resolver: Optional[RetrievalFewShotResolver],
        k_shot: int,
    ) -> List[List[FewShotExample]]:
        """Resolve retrieval few-shots for each example in a batch."""
        if resolver is None or k_shot == 0:
            return [[] for _ in batch_examples]

        batch_few_shots: List[List[FewShotExample]] = []
        for example in batch_examples:
            query_dataset = str(getattr(example, "source_dataset", dataset_name)).strip().lower()
            resolved = resolver.resolve(
                query_dataset=query_dataset,
                query_id=example.example_id,
                k_shot=k_shot,
            )
            batch_few_shots.append(resolved)
        return batch_few_shots

    def _write_batch_records(
        self,
        *,
        writer: TextIO,
        batch_examples: Sequence[DatasetExample],
        batch_few_shots: Sequence[Sequence[FewShotExample]],
        batch_result: BatchResult,
        dataset_name: str,
        model_spec: ModelSpec,
        quant_mode: str,
        loaded_engine: LoadedEngine,
        config: RunConfig,
        few_shot_details: Dict[str, Any],
        reasoning_enabled: bool,
        parse_stats: ParseStats,
    ) -> None:
        for idx_in_batch, example in enumerate(batch_examples):
            payload = batch_result.example_records[idx_in_batch]
            valid, repaired = batch_result.parse_flags[idx_in_batch]
            overflow_flag = batch_result.overflow_flags[idx_in_batch]
            few_shot_count = len(batch_few_shots[idx_in_batch])

            final_record = self._build_final_record(
                dataset_name=dataset_name,
                example=example,
                model_spec=model_spec,
                quant_mode=quant_mode,
                loaded_engine=loaded_engine,
                config=config,
                few_shot_details=few_shot_details,
                reasoning_enabled=reasoning_enabled,
                few_shot_count=few_shot_count,
                payload=payload,
                batch_size=len(batch_examples),
                tokens_per_second=batch_result.tokens_per_second,
                parse_valid=valid,
                parse_repaired=repaired,
                overflow_flag=overflow_flag,
            )
            writer.write(json.dumps(final_record, separators=(",", ":")) + "\n")

            parse_stats.success += int(valid)
            parse_stats.failed += int(not valid)
            parse_stats.repaired += int(repaired)
            parse_stats.overflow_skipped += int(overflow_flag)

    def _build_final_record(
        self,
        *,
        dataset_name: str,
        example: DatasetExample,
        model_spec: ModelSpec,
        quant_mode: str,
        loaded_engine: LoadedEngine,
        config: RunConfig,
        few_shot_details: Dict[str, Any],
        reasoning_enabled: bool,
        few_shot_count: int,
        payload: Dict[str, Any],
        batch_size: int,
        tokens_per_second: float,
        parse_valid: bool,
        parse_repaired: bool,
        overflow_flag: bool,
    ) -> Dict[str, Any]:
        """Materialize one output JSONL row with stable metadata keys."""
        return {
            "dataset": dataset_name,
            "example_id": example.example_id,
            "source_dataset": example.source_dataset,
            "provider": model_spec.provider,
            "model_name": model_spec.model_name,
            "quant_mode": quant_mode,
            "quant_method": loaded_engine.quant_method,
            "quant_repo": loaded_engine.repo_id,
            "quant_revision": loaded_engine.revision or "main",
            "batch_size": batch_size,
            "k_shot": config.k_shot,
            "seed": config.seed,
            "input_context": config.input_context,
            "base_prompt": config.base_prompt,
            "prompt_variant": self.prompt_builder.prompt_variant_label,
            "few_shot_preset": config.few_shot_preset,
            "few_shot_mode": few_shot_details["mode"],
            "few_shot_archive_split": few_shot_details["split"],
            "few_shot_archive_path": few_shot_details["archive"],
            "reasoning_enabled": reasoning_enabled,
            "few_shot_examples": few_shot_count,
            "latency_ms": payload["latency_ms"],
            "input_tokens": payload["input_tokens"],
            "output_tokens": payload["output_tokens"],
            "report_only_input_tokens": payload["report_only_input_tokens"],
            "throughput_toks_per_sec": tokens_per_second,
            "labels_pred": payload["labels_pred"],
            "labels_gold": example.labels_four_class,
            "raw_response": payload["raw_response"],
            "reasoning": payload["reasoning"],
            "overflow": overflow_flag,
            "parse_valid": parse_valid,
            "parse_repaired": parse_repaired,
        }

    @staticmethod
    def _accumulate_batch_metrics(metrics: RunMetrics, batch_result: BatchResult) -> None:
        metrics.total_latency += batch_result.latency_ms
        metrics.total_input_tokens += sum(batch_result.input_token_counts)
        metrics.total_output_tokens += sum(batch_result.output_token_counts)
        metrics.total_report_only_tokens += sum(batch_result.report_token_counts)

    def _log_run_summary(
        self,
        *,
        log: LogFn,
        total_examples: int,
        preds_path: Path,
        metrics: RunMetrics,
    ) -> None:
        """Emit aggregate throughput/parse metrics at end of run."""
        parse_stats = metrics.parse_stats
        avg_latency = metrics.total_latency / max(1, total_examples)
        avg_input_tokens = metrics.total_input_tokens / max(1, total_examples)
        avg_output_tokens = metrics.total_output_tokens / max(1, total_examples)
        overall_throughput = metrics.total_output_tokens / max(metrics.total_latency / 1000.0, 1e-6)
        elapsed_minutes = max(metrics.total_latency / 60000.0, 1e-6)
        reports_per_minute = total_examples / elapsed_minutes
        prompt_tokens_per_minute = metrics.total_input_tokens / elapsed_minutes
        report_tokens_per_minute = metrics.total_report_only_tokens / elapsed_minutes

        log(
            f"Completed inference: latency_avg_ms={avg_latency:.2f} "
            f"input_tokens_avg={avg_input_tokens:.2f} "
            f"output_tokens_avg={avg_output_tokens:.2f}"
        )
        log(f"Throughput tokens/sec={overall_throughput:.2f}")
        log(
            f"Throughput reports/min={reports_per_minute:.2f} "
            f"prompt_tokens/min={prompt_tokens_per_minute:.2f} "
            f"report_tokens/min={report_tokens_per_minute:.2f}"
        )
        log(
            "Parse stats: success=%d repaired=%d failed=%d overflow_skipped=%d"
            % (
                parse_stats.success,
                parse_stats.repaired,
                parse_stats.failed,
                parse_stats.overflow_skipped,
            )
        )
        if parse_stats.total_attempted():
            violation_rate = parse_stats.failed / parse_stats.total_attempted()
            repair_rate = parse_stats.repaired / parse_stats.total_attempted()
            log(f"Schema violation rate={violation_rate:.4f} repair rate={repair_rate:.4f}")
        if metrics.auto_reduce_count:
            log(f"auto_reduce_on_oom triggered {metrics.auto_reduce_count} times.")
        if torch is not None and torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            log(f"Peak CUDA memory allocated={peak_memory_mb:.2f} MB")

        log("Prediction file written to %s" % preds_path)

    @staticmethod
    def _score_run(
        *,
        log: LogFn,
        preds_path: Path,
        dataset_name: str,
        results_root: Path,
    ) -> None:
        """Best-effort hook for automatic scoring when the optional module is present."""
        try:
            from . import scoring as scoring_module  # type: ignore
        except ImportError:
            log("Scoring module not available; skipping automatic metrics computation.")
            return

        try:
            scoring_module.score_run(  # type: ignore[attr-defined]
                preds_path=preds_path,
                dataset=dataset_name,
                outdir=results_root,
            )
        except AttributeError:
            log("scoring.score_run not implemented; run manual scoring once available.")

    def _apply_prompt_overrides(self, config: RunConfig) -> None:
        """Configure the prompt builder based on run-level overrides."""
        if config.custom_system_prompt is None and config.prompt_variant is None:
            self.prompt_builder = self._default_prompt_builder
            return

        self.prompt_builder = self._build_prompt_builder(
            prompt_variant=config.prompt_variant,
            custom_system_prompt=config.custom_system_prompt,
        )

    def _resolve_few_shot_retrieval(
        self,
        *,
        config: RunConfig,
        dataset_name: str,
    ) -> Tuple[Optional[RetrievalFewShotResolver], Dict[str, Any]]:
        details: Dict[str, Any] = {
            "mode": "zero-shot",
            "split": "none",
            "archive": "none",
            "pool": str(config.few_shot_pool_path),
            "strict": bool(config.strict_few_shot_lookup),
        }
        if config.k_shot == 0:
            return None, details

        split = config.few_shot_archive_split or self._infer_retrieval_split(dataset_name)
        if split is None:
            raise ValueError(
                "Unable to infer retrieval archive split for k-shot run. "
                "Set RunConfig.few_shot_archive_split to 'dev' or 'test'."
            )
        resolver = RetrievalFewShotResolver(
            split=split,
            archive_path=config.few_shot_archive_path,
            pool_path=config.few_shot_pool_path,
            strict=config.strict_few_shot_lookup,
        )
        details.update(
            {
                "mode": f"retrieval-{resolver.split}",
                "split": resolver.split,
                "archive": str(resolver.archive_path),
                "pool": str(resolver.pool_path),
                "strict": resolver.strict,
            }
        )
        return resolver, details

    @staticmethod
    def _infer_retrieval_split(dataset_name: str) -> Optional[str]:
        normalized = str(dataset_name).strip().lower()
        if normalized.startswith("dev") or normalized.endswith("_dev"):
            return "dev"
        if normalized.startswith("test") or normalized.endswith("_test"):
            return "test"
        return None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _generate_batch(
        self,
        *,
        batch_examples: Sequence[DatasetExample],
        loaded_engine: LoadedEngine,
        gen_config: GenerationConfig,
        prompt_seed: Optional[int],
        use_structured_output: bool,
        few_shots_by_example: Sequence[Sequence[FewShotExample]],
    ) -> BatchResult:
        if len(few_shots_by_example) != len(batch_examples):
            raise ValueError(
                "few_shots_by_example length must match batch_examples length "
                f"({len(few_shots_by_example)} != {len(batch_examples)})."
            )

        messages: List[List[Dict[str, str]]] = []
        for idx, example in enumerate(batch_examples):
            shots_for_example = list(few_shots_by_example[idx])
            messages.append(
                self.prompt_builder.build_messages(
                    report_text=example.text,
                    report_id=example.example_id,
                    few_shots=shots_for_example,
                )
            )
        use_harmony = self._should_use_harmony(loaded_engine.reasoning_parser, gen_config)
        if use_harmony:
            prompt_token_ids_batch = self._render_harmony_prompts(
                messages_batch=messages,
                reasoning_effort=gen_config.reasoning_effort,
            )
            prompts = [{"prompt_token_ids": token_ids} for token_ids in prompt_token_ids_batch]
            prompt_token_counts = [len(token_ids) for token_ids in prompt_token_ids_batch]
        else:
            chat_texts = self._render_chat_texts(
                model_name=loaded_engine.model_spec.model_name,
                tokenizer=loaded_engine.tokenizer,
                messages_batch=messages,
            )
            prompts = chat_texts
            tokenized = loaded_engine.tokenizer(
                chat_texts,
                padding=False,
                return_length=True,
            )
            prompt_token_counts = [int(length) for length in tokenized["length"]]
        report_encodings = loaded_engine.tokenizer(
            [example.text for example in batch_examples],
            padding=False,
            add_special_tokens=False,
            return_length=True,
        )
        report_token_counts = [int(length) for length in report_encodings["length"]]

        sampling_params = self._make_sampling_params(
            gen_config,
            prompt_seed,
            use_structured_output=use_structured_output,
        )
        start = time.perf_counter()
        outputs = loaded_engine.engine.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        max_model_len = self._max_model_len(loaded_engine)

        records: List[Dict[str, Any]] = []
        parse_flags: List[Tuple[bool, bool]] = []
        overflow_flags: List[bool] = []
        raw_outputs: List[str] = []
        reasoning_outputs: List[Optional[str]] = []
        output_token_counts: List[int] = []

        total_output_tokens = 0
        for example, request_output, prompt_tokens, report_tokens in zip(
            batch_examples,
            outputs,
            prompt_token_counts,
            report_token_counts,
        ):
            sequence_output = request_output.outputs[0]
            raw_text = sequence_output.text
            reasoning_text, content_text = self._extract_reasoning_content(
                raw_text=raw_text,
                token_ids=sequence_output.token_ids,
                reasoning_parser=loaded_engine.reasoning_parser,
            )
            text_for_parsing = content_text if content_text is not None else raw_text
            completion_tokens = int(len(sequence_output.token_ids))
            prompt_tokens = int(getattr(request_output, "prompt_token_count", prompt_tokens))
            finish_reason = getattr(sequence_output, "finish_reason", None)
            finish_reason_text = str(
                getattr(finish_reason, "value", getattr(finish_reason, "name", finish_reason))
            ).lower()
            overflow_flag = (
                (max_model_len is not None and prompt_tokens >= max_model_len)
                or finish_reason_text == FINISH_REASON_LENGTH
            )

            parsed_labels, valid, repaired = self._parse_model_output(text_for_parsing)
            record = {
                "labels_pred": parsed_labels,
                "input_tokens": int(prompt_tokens),
                "output_tokens": completion_tokens,
                "report_only_input_tokens": int(report_tokens),
                "latency_ms": latency_ms / max(1, len(batch_examples)),
                "raw_response": raw_text,
                "reasoning": reasoning_text,
                "overflow": bool(overflow_flag),
            }
            records.append(record)
            parse_flags.append((valid, repaired))
            overflow_flags.append(bool(overflow_flag))
            raw_outputs.append(raw_text)
            reasoning_outputs.append(reasoning_text)
            output_token_counts.append(completion_tokens)
            total_output_tokens += completion_tokens

        tokens_per_second = total_output_tokens / max(latency_ms / 1000.0, 1e-6)

        return BatchResult(
            example_records=records,
            raw_outputs=raw_outputs,
            reasoning_outputs=reasoning_outputs,
            input_token_counts=[int(val) for val in prompt_token_counts],
            output_token_counts=output_token_counts,
            report_token_counts=report_token_counts,
            latency_ms=latency_ms,
            tokens_per_second=tokens_per_second,
            parse_flags=parse_flags,
            overflow_flags=overflow_flags,
        )

    def _render_chat_texts(
        self,
        *,
        model_name: str,
        tokenizer: Any,
        messages_batch: Sequence[Sequence[Dict[str, str]]],
    ) -> List[str]:
        mode = self._chat_template_mode_by_model.get(model_name, "default")
        if mode == "user_assistant_only":
            messages_batch = [
                self._to_user_assistant_messages(messages)
                for messages in messages_batch
            ]

        try:
            return [
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                for messages in messages_batch
            ]
        except Exception as exc:
            if mode != "default" or not self._is_role_alternation_error(exc):
                raise

            converted_batch = [
                self._to_user_assistant_messages(messages)
                for messages in messages_batch
            ]
            rendered = [
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                for messages in converted_batch
            ]
            self._chat_template_mode_by_model[model_name] = "user_assistant_only"
            print(
                f"[infer] Chat template for model '{model_name}' requires alternating user/assistant roles. "
                "Folding system prompts into the first user message."
            )
            return rendered

    @staticmethod
    def _is_role_alternation_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "must alternate user/assistant" in message
            or "alternate user/assistant" in message
            or "only user and assistant roles are supported" in message
        )

    @staticmethod
    def _to_user_assistant_messages(messages: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
        system_chunks: List[str] = []
        converted: List[Dict[str, str]] = []

        for message in messages:
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", ""))
            if role == "system":
                if content.strip():
                    system_chunks.append(content.strip())
                continue
            if role not in {"user", "assistant"}:
                role = "user"
            converted.append({"role": role, "content": content})

        if system_chunks:
            system_prefix = "\n\n".join(system_chunks)
            if converted and converted[0]["role"] == "user":
                first_content = converted[0].get("content", "")
                converted[0]["content"] = f"{system_prefix}\n\n{first_content}".strip()
            else:
                converted.insert(0, {"role": "user", "content": system_prefix})

        if not converted:
            return [{"role": "user", "content": ""}]
        return converted

    def _max_model_len(self, loaded_engine: LoadedEngine) -> Optional[int]:
        engine_inner = getattr(loaded_engine.engine, "llm_engine", None)
        if engine_inner is not None:
            model_config = getattr(engine_inner, "model_config", None)
            if model_config is not None:
                return getattr(model_config, "max_model_len", None)
        return loaded_engine.engine_options.get("max_model_len")

    def _is_recoverable_oom(self, err: BaseException) -> bool:
        message = str(err).lower()
        return any(keyword in message for keyword in OOM_KEYWORDS)

    def _handle_oom_recovery(
        self,
        *,
        log: LogFn,
        previous_engine: LoadedEngine,
        current_batch_size: int,
        active_overrides: Dict[str, Any],
    ) -> Tuple[LoadedEngine, int, Dict[str, Any]]:
        if current_batch_size <= 1:
            raise RuntimeError(
                "vLLM reported OOM with batch_size=1; manual intervention required."
            )

        new_batch_size = max(1, current_batch_size // 2)
        log(f"OOM encountered; reducing batch_size from {current_batch_size} to {new_batch_size}.")

        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return previous_engine, new_batch_size, dict(active_overrides)

    def _build_json_schema(self) -> Dict[str, Any]:
        label_properties: Dict[str, Any] = {}
        for canonical in self.label_order:
            title_label = self._canonical_to_title[canonical]
            if canonical == "no_finding":
                label_properties[title_label] = {"type": "integer", "enum": [0, 1]}
            else:
                label_properties[title_label] = {
                    "type": ["integer", "null"],
                    "enum": [-1, 0, 1, None],
                }

        labels_object = {
            "type": "object",
            "properties": label_properties,
            "required": self.title_label_order,
            "additionalProperties": False,
        }

        return {
            "type": "object",
            "properties": {
                "labels": labels_object,
            },
            "required": ["labels"],
            "additionalProperties": False,
        }

    def _make_sampling_params(
        self,
        gen_config: GenerationConfig,
        seed: Optional[int],
        *,
        use_structured_output: bool,
    ) -> SamplingParams:
        top_k = gen_config.top_k if gen_config.top_k is not None else -1
        extra_kwargs: Dict[str, Any] = {}
        if use_structured_output and StructuredOutputsParams is not None:
            extra_kwargs["structured_outputs"] = StructuredOutputsParams(
                json=self._json_schema
            )
        return SamplingParams(
            temperature=gen_config.temperature,
            top_p=gen_config.top_p,
            top_k=top_k,
            max_tokens=gen_config.max_new_tokens,
            repetition_penalty=gen_config.repetition_penalty,
            seed=seed,
            **extra_kwargs,
        )

    @staticmethod
    def _should_use_harmony(reasoning_parser: Optional[str], gen_config: GenerationConfig) -> bool:
        return reasoning_parser == "openai_gptoss" and bool(gen_config.reasoning_effort)

    @staticmethod
    def _render_harmony_prompts(
        *,
        messages_batch: Sequence[Sequence[Dict[str, str]]],
        reasoning_effort: Optional[str],
    ) -> List[List[int]]:
        from vllm.entrypoints.harmony_utils import (
            get_developer_message,
            get_system_message,
            parse_chat_input,
            render_for_completion,
        )

        prompt_token_ids_batch: List[List[int]] = []
        for messages in messages_batch:
            harmony_messages = [
                get_system_message(
                    reasoning_effort=reasoning_effort,
                    browser_description=None,
                    python_description=None,
                    container_description=None,
                ),
                get_developer_message(tools=None),
            ]
            for chat_msg in messages:
                harmony_messages.extend(parse_chat_input(chat_msg))
            prompt_token_ids_batch.append(render_for_completion(harmony_messages))
        return prompt_token_ids_batch

    def _extract_reasoning_content(
        self,
        *,
        raw_text: str,
        token_ids: Sequence[int],
        reasoning_parser: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        if not reasoning_parser:
            return None, None

        if reasoning_parser == "qwen3":
            start_token = "<think>"
            end_token = "</think>"
            start_idx = raw_text.find(start_token)
            if start_idx != -1:
                end_idx = raw_text.find(end_token, start_idx + len(start_token))
                if end_idx != -1:
                    reasoning = raw_text[start_idx + len(start_token) : end_idx].strip()
                    content = (raw_text[:start_idx] + raw_text[end_idx + len(end_token) :]).strip()
                    return (reasoning or None, content or None)
            return None, None

        if reasoning_parser == "openai_gptoss":
            try:
                from vllm.reasoning.gptoss_reasoning_parser import parse_chat_output

                reasoning, content, _ = parse_chat_output(list(token_ids))
            except Exception:
                return None, None
            reasoning = reasoning.strip() if reasoning else None
            content = content.strip() if content else None
            return reasoning, content

        return None, None

    def _parse_model_output(
        self,
        text: str,
    ) -> Tuple[Dict[str, Optional[int]], bool, bool]:
        cleaned = text.strip()
        parse_attempts: List[str] = [cleaned]
        extracted = extract_first_json_object(cleaned)
        if extracted and extracted != cleaned:
            parse_attempts.append(extracted)

        parsed_obj: Optional[Dict[str, Any]] = None
        repaired = False
        for attempt_idx, candidate in enumerate(parse_attempts):
            try:
                maybe = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(maybe, dict):
                parsed_obj = maybe
                repaired = attempt_idx > 0
                break

        if parsed_obj is None or "labels" not in parsed_obj or not isinstance(parsed_obj["labels"], dict):
            fallback = self._fallback_labels()
            return fallback, False, repaired

        labels_raw: Dict[str, Any] = parsed_obj["labels"]
        normalized = self._normalize_labels(labels_raw)
        return normalized, True, repaired

    def _normalize_labels(
        self,
        labels_raw: Dict[str, Any],
    ) -> Dict[str, Optional[int]]:
        normalized: Dict[str, Optional[int]] = {}

        canonical_inputs: Dict[str, Any] = {}
        for raw_key, raw_value in labels_raw.items():
            canonical_key: Optional[str] = None
            if raw_key in self._title_to_canonical:
                canonical_key = self._title_to_canonical[raw_key]
            elif raw_key in self.label_order:
                canonical_key = raw_key
            elif isinstance(raw_key, str):
                stripped = raw_key.strip()
                if stripped in self._title_to_canonical:
                    canonical_key = self._title_to_canonical[stripped]
                elif stripped in self.label_order:
                    canonical_key = stripped
            if canonical_key is None:
                continue
            canonical_inputs[canonical_key] = raw_value

        findings: Dict[str, Optional[int]] = {}
        for key in self.label_order:
            findings[key] = self._normalize_value(canonical_inputs.get(key))

        normalized.update(findings)
        return {key: normalized.get(key) for key in self.label_order}

    def _normalize_value(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, (int, float)):
            ivalue = int(value)
            if ivalue in {1, 0, -1}:
                return ivalue
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            lowered = stripped.lower()
            if lowered in {"null", "none"}:
                return None
            if lowered in {"1", "+1", "true", "present"}:
                return 1
            if lowered in {"0", "-0", "false", "absent"}:
                return 0
            if lowered in {"-1", "uncertain"}:
                return -1
        return None

    def _fallback_labels(self) -> Dict[str, Optional[int]]:
        placeholder = {key: None for key in self.label_order if key != "no_finding"}
        combined = {"no_finding": 0}
        combined.update(placeholder)
        return {key: combined.get(key) for key in self.label_order}


__all__ = ["InferenceRunner", "RunConfig", "GenerationConfig"]
