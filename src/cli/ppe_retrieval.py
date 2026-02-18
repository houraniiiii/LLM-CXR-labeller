"""Unified retrieval-aware CLI for PPE runs across one dataset or all datasets."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

from ..core.engines import ModelSpec, VLLMEngineRegistry
from ..core.infer import GenerationConfig, InferenceRunner, RunConfig
from ..core.prompts import PromptVariant

RUN_DEFAULTS_PATH = Path("configs") / "run_defaults.yaml"


def load_defaults(path: Path = RUN_DEFAULTS_PATH) -> dict:
    data: dict = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

    defaults = data.get("defaults", {}) or {}
    few_shot_cfg = data.get("few_shot", {}) or {}
    prompt_cfg = data.get("prompts", {}) or {}
    batching_cfg = data.get("batching", {}) or {}
    vllm_cfg = data.get("vllm", {}) or {}
    valid_prompts = {member.value for member in PromptVariant}
    default_prompt_variant = str(
        prompt_cfg.get("default_variant", PromptVariant.CLINICAL_STANDARD.value)
    )
    if default_prompt_variant not in valid_prompts:
        default_prompt_variant = PromptVariant.CLINICAL_STANDARD.value

    allowed_raw = few_shot_cfg.get("allowed_shots", [0, 5, 10])
    allowed_shots: List[int] = []
    for value in allowed_raw:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed in {0, 5, 10} and parsed not in allowed_shots:
            allowed_shots.append(parsed)
    if 0 not in allowed_shots:
        allowed_shots.insert(0, 0)
    if not allowed_shots:
        allowed_shots = [0, 5, 10]

    def _coerce_int(value):
        return None if value is None else int(value)

    def _coerce_float(value):
        return None if value is None else float(value)

    return {
        "seed": int(defaults.get("seed", 42)),
        "k_shot": int(defaults.get("k_shot", 0)),
        "max_new_tokens": int(defaults.get("max_new_tokens", 2000)),
        "outdir": Path(defaults.get("outdir", "results")),
        "limit": defaults.get("limit"),
        "allowed_shots": allowed_shots,
        "default_prompt_variant": default_prompt_variant,
        "auto_reduce_on_oom": bool(batching_cfg.get("auto_reduce_on_oom", True)),
        "vllm": {
            "tensor_parallel_size": _coerce_int(vllm_cfg.get("tensor_parallel_size")),
            "gpu_memory_utilization": _coerce_float(vllm_cfg.get("gpu_memory_utilization")),
            "max_num_batched_tokens": _coerce_int(vllm_cfg.get("max_num_batched_tokens")),
            "swap_space_gb": _coerce_int(vllm_cfg.get("swap_space_gb")),
            "kv_cache_dtype": vllm_cfg.get("kv_cache_dtype"),
        },
    }


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def available_prompt_variants() -> List[str]:
    return [member.value for member in PromptVariant]


def resolve_prompt_variants(
    args: argparse.Namespace,
    *,
    default_prompt_variant: str,
) -> List[str]:
    if args.all_prompts:
        return available_prompt_variants()
    if args.prompt_variants:
        return dedupe_preserve_order(args.prompt_variants)
    return [default_prompt_variant]


def select_model_specs(registry: VLLMEngineRegistry, args: argparse.Namespace) -> List[ModelSpec]:
    specs: List[ModelSpec] = []

    if args.all_models:
        specs.extend(registry.list_specs())

    if args.provider:
        for provider in args.provider:
            provider_specs = registry.list_specs(provider=provider)
            if not provider_specs:
                raise ValueError(f"No models found for provider '{provider}'.")
            specs.extend(provider_specs)

    if args.model:
        for model_name in args.model:
            try:
                specs.append(registry.get(model_name))
            except KeyError as err:
                raise ValueError(str(err)) from err

    if not specs:
        raise ValueError("Specify --model, --provider, or --all-models.")

    ordered = dedupe_preserve_order(spec.model_name for spec in specs)
    return [registry.get(name) for name in ordered]


def build_parser(defaults: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run retrieval-aware PPE inference on dev/test splits using MedCPT "
            "few-shot archives (k-shot in {0,5,10})."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "indiana", "mimic", "rexgradient"],
        default="all",
        help="Target dataset. 'all' uses dev_all/test_all combined files.",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="dev",
        help="Evaluation split and retrieval archive to use.",
    )
    parser.add_argument(
        "--quant-mode",
        choices=["fp16", "int4"],
        default="fp16",
        help="Precision mode for model loading.",
    )
    parser.add_argument(
        "--model",
        action="append",
        metavar="MODEL_ID",
        help="Exact Hugging Face model repo to run (may be passed multiple times).",
    )
    parser.add_argument(
        "--provider",
        action="append",
        metavar="PROVIDER",
        help="Select all models for a provider (case-insensitive exact match).",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run every model listed in models.jsonl for this precision.",
    )
    parser.add_argument(
        "--k-shot",
        type=int,
        choices=sorted(defaults["allowed_shots"]),
        default=defaults["k_shot"],
        help="Few-shot count from retrieval archive ranking.",
    )
    parser.add_argument(
        "--prompt-variants",
        nargs="+",
        metavar="PROMPT",
        choices=available_prompt_variants(),
        help=(
            f"Prompt variant(s) to evaluate. Defaults to {defaults['default_prompt_variant']}; "
            "use --all-prompts to run all."
        ),
    )
    parser.add_argument(
        "--all-prompts",
        action="store_true",
        help="Evaluate all built-in prompt variants in one invocation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override default batch size from models.jsonl.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=defaults["max_new_tokens"],
        help="Maximum tokens to generate per report.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default=None,
        help="Set GPT-OSS reasoning effort (Harmony) when supported.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults["seed"],
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=defaults["limit"],
        help="Limit the number of examples (for smoke tests).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=defaults["outdir"],
        help="Output directory root for predictions, metrics, and logs.",
    )
    parser.add_argument(
        "--disable-auto-reduce",
        action="store_true",
        help="Disable automatic batch-size reduction on CUDA OOM.",
    )
    parser.add_argument(
        "--suppress-reasoning",
        action="store_true",
        help="Force structured-output decoding even when reasoning parsers are available.",
    )
    parser.add_argument(
        "--allow-missing-retrieval",
        action="store_true",
        help="Allow missing retrieval keys; missing examples fall back to zero-shot.",
    )

    vllm_defaults = defaults["vllm"]
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=vllm_defaults["tensor_parallel_size"],
        help="Tensor parallelism degree for vLLM.",
    )
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=vllm_defaults["gpu_memory_utilization"],
        help="Target GPU memory utilisation for vLLM (0-1).",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=vllm_defaults["max_num_batched_tokens"],
        help="Maximum total batched tokens per vLLM step.",
    )
    parser.add_argument(
        "--swap-space-gb",
        type=int,
        default=vllm_defaults["swap_space_gb"],
        help="Host swap space to reserve for vLLM (GiB).",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        default=vllm_defaults["kv_cache_dtype"],
        help="Override vLLM KV cache dtype (e.g., fp8, fp16, auto).",
    )
    return parser


def _dataset_name(dataset: str, split: str) -> str:
    if dataset == "all":
        return f"{split}_all"
    return f"{dataset}_{split}"


def _vllm_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if args.tensor_parallel is not None:
        overrides["tensor_parallel_size"] = args.tensor_parallel
    if args.gpu_mem_util is not None:
        overrides["gpu_memory_utilization"] = args.gpu_mem_util
    if args.max_num_batched_tokens is not None:
        overrides["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.swap_space_gb is not None:
        overrides["swap_space_gb"] = args.swap_space_gb
    if args.kv_cache_dtype:
        overrides["kv_cache_dtype"] = args.kv_cache_dtype
    return overrides


def make_run_config(
    *,
    args: argparse.Namespace,
    model_name: str,
    prompt_variant: str,
) -> RunConfig:
    return RunConfig(
        model_name=model_name,
        dataset=_dataset_name(args.dataset, args.split),
        input_context="full",
        quant_mode=args.quant_mode,
        base_prompt=None,
        k_shot=args.k_shot,
        requested_k_shot=args.k_shot,
        prompt_variant=prompt_variant,
        custom_system_prompt=None,
        few_shot_preset=None,
        few_shot_archive_split=args.split,
        batch_size=args.batch_size,
        seed=args.seed,
        limit=args.limit,
        outdir=args.outdir,
        auto_reduce_on_oom=not args.disable_auto_reduce,
        generation=GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            reasoning_effort=args.reasoning_effort,
        ),
        vllm_options=_vllm_overrides(args),
        suppress_reasoning=args.suppress_reasoning,
        strict_few_shot_lookup=not args.allow_missing_retrieval,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    defaults = load_defaults()
    parser = build_parser(defaults)
    args = parser.parse_args(argv)

    prompt_variants = resolve_prompt_variants(
        args,
        default_prompt_variant=defaults["default_prompt_variant"],
    )
    registry = VLLMEngineRegistry()
    runner = InferenceRunner(model_registry=registry)
    try:
        specs = select_model_specs(registry, args)
    except ValueError as exc:
        parser.error(str(exc))

    for spec in specs:
        print(
            f"[ppe_retrieval] Preparing runs for model: {spec.model_name} "
            f"(provider={spec.provider})"
        )
        try:
            for prompt_variant in prompt_variants:
                print(
                    f"[ppe_retrieval] Running model: {spec.model_name} "
                    f"dataset={args.dataset} split={args.split} quant={args.quant_mode} "
                    f"prompt={prompt_variant} k={args.k_shot}"
                )
                config = make_run_config(
                    args=args,
                    model_name=spec.model_name,
                    prompt_variant=prompt_variant,
                )
                runner.run(config)
        finally:
            evicted = registry.evict_model(spec.model_name)
            if evicted > 0:
                print(
                    f"[ppe_retrieval] Released {evicted} cached engine(s) for model: {spec.model_name}"
                )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
