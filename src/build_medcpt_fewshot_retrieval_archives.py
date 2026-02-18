from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


DEFAULT_FEW_SHOT_PATH = Path("data/embeddings/few_shot_50_medcpt_article.jsonl")
DEFAULT_DEV_PATH = Path("data/embeddings/dev_all_medcpt_article.jsonl")
DEFAULT_TEST_PATH = Path("data/embeddings/test_all_medcpt_article.jsonl")
DEFAULT_OUT_DIR = Path("data/embeddings/retrieval_archives")
DEFAULT_TOP_K = 10
DEFAULT_EXPECTED_MODEL = "ncbi/MedCPT-Article-Encoder"


@dataclass(frozen=True)
class EmbeddingTable:
    records: List[dict]
    matrix: np.ndarray
    model_name: str
    model_revision: str
    keys: Set[str]


def _as_key(dataset: str, example_id: str) -> str:
    return f"{dataset.strip().lower()}::{example_id.strip()}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build top-k few-shot retrieval archives from embedding JSONL files."
    )
    parser.add_argument("--few-shot-path", type=Path, default=DEFAULT_FEW_SHOT_PATH)
    parser.add_argument("--dev-path", type=Path, default=DEFAULT_DEV_PATH)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--expected-model",
        type=str,
        default=DEFAULT_EXPECTED_MODEL,
        help="Fail if embedding_model in source rows differs.",
    )
    parser.add_argument(
        "--allow-query-few-overlap",
        action="store_true",
        help="Allow overlap between query keys and few-shot keys (disabled by default).",
    )
    return parser.parse_args()


def load_embedding_jsonl(path: Path, *, expected_model: Optional[str]) -> EmbeddingTable:
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    records: List[dict] = []
    vectors: List[np.ndarray] = []
    key_set: Set[str] = set()
    embedding_models: Set[str] = set()
    embedding_revisions: Set[str] = set()
    dim: Optional[int] = None

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc.msg}") from exc

            if not isinstance(payload, dict):
                raise ValueError(f"Row {line_no} in {path} must be a JSON object.")

            dataset = str(payload.get("dataset", "")).strip().lower()
            example_id = str(payload.get("id", "")).strip()
            if not dataset or not example_id:
                raise ValueError(f"Row {line_no} in {path} missing dataset/id.")

            model_name = str(payload.get("embedding_model", "")).strip()
            if expected_model and model_name and model_name != expected_model:
                raise ValueError(
                    f"Row {line_no} in {path} has embedding_model={model_name!r}; "
                    f"expected {expected_model!r}."
                )
            embedding_models.add(model_name or "unknown")
            embedding_revisions.add(str(payload.get("embedding_revision", "")).strip() or "unknown")

            key = _as_key(dataset, example_id)
            if key in key_set:
                raise ValueError(f"Duplicate key {key!r} found in {path} (line {line_no}).")

            if "embedding" not in payload:
                raise ValueError(f"Row {line_no} in {path} missing 'embedding' field.")
            emb = np.asarray(payload["embedding"], dtype=np.float32)
            if emb.ndim != 1 or emb.size == 0:
                raise ValueError(
                    f"Row {line_no} in {path} has invalid embedding shape {emb.shape!r}; expected 1-D vector."
                )
            if dim is None:
                dim = int(emb.shape[0])
            elif int(emb.shape[0]) != dim:
                raise ValueError(
                    f"Embedding dimension mismatch in {path} line {line_no}: got {emb.shape[0]}, expected {dim}."
                )

            records.append(
                {
                    "dataset": dataset,
                    "id": example_id,
                    "key": key,
                    "line_number": line_no,
                }
            )
            key_set.add(key)
            vectors.append(emb)

    if not vectors:
        raise ValueError(f"No embeddings found in {path}")

    if len(embedding_models) != 1:
        raise ValueError(f"Found multiple embedding_model values in {path}: {sorted(embedding_models)}")

    matrix = np.vstack(vectors)
    model_name = next(iter(embedding_models))
    model_revision = (
        next(iter(embedding_revisions)) if len(embedding_revisions) == 1 else "mixed"
    )
    return EmbeddingTable(
        records=records,
        matrix=matrix,
        model_name=model_name,
        model_revision=model_revision,
        keys=key_set,
    )


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return matrix / norms


def compute_topk_lookup(
    query_table: EmbeddingTable,
    query_mat: np.ndarray,
    few_table: EmbeddingTable,
    few_mat: np.ndarray,
    top_k: int,
) -> Dict[str, List[dict]]:
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0; received {top_k}.")

    sims = query_mat @ few_mat.T
    k = min(top_k, few_mat.shape[0])
    lookup: Dict[str, List[dict]] = {}

    for i, query in enumerate(query_table.records):
        row = sims[i]
        candidate_idx = np.argpartition(-row, kth=k - 1)[:k] if k < len(row) else np.arange(len(row))
        ordered_idx = sorted(
            candidate_idx.tolist(),
            key=lambda idx: (
                -float(row[int(idx)]),
                few_table.records[int(idx)]["dataset"],
                few_table.records[int(idx)]["id"],
            ),
        )
        key = query["key"]
        ranked: List[dict] = []
        for rank, idx in enumerate(ordered_idx, start=1):
            candidate = few_table.records[int(idx)]
            ranked.append(
                {
                    "rank": rank,
                    "few_shot_dataset": candidate["dataset"],
                    "few_shot_id": candidate["id"],
                    "few_shot_line_number": candidate["line_number"],
                    "cosine_similarity": float(row[int(idx)]),
                }
            )
        lookup[key] = ranked
    return lookup


def write_archive_jsonl(
    output_path: Path,
    *,
    split_name: str,
    query_source: Path,
    few_shot_source: Path,
    embedding_model: str,
    embedding_revision: str,
    embedding_dim: int,
    lookup: Dict[str, List[dict]],
    top_k: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for key in sorted(lookup.keys()):
            query_dataset, query_id = key.split("::", 1)
            payload = {
                "split": split_name,
                "embedding_model": embedding_model,
                "embedding_revision": embedding_revision,
                "embedding_dim": embedding_dim,
                "query_source": str(query_source),
                "few_shot_source": str(few_shot_source),
                "top_k": top_k,
                "query_dataset": query_dataset,
                "query_id": query_id,
                "matches": lookup[key],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError(f"--top-k must be > 0; received {args.top_k}.")

    few_table = load_embedding_jsonl(args.few_shot_path, expected_model=args.expected_model)
    if few_table.model_name == "unknown":
        raise ValueError(
            f"{args.few_shot_path} is missing embedding_model metadata; regenerate embeddings with metadata fields."
        )
    few_mat = few_table.matrix
    few_mat = l2_normalize(few_mat)

    dev_table = load_embedding_jsonl(args.dev_path, expected_model=args.expected_model)
    if few_table.matrix.shape[1] != dev_table.matrix.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch few-shot/dev: {few_table.matrix.shape[1]} vs {dev_table.matrix.shape[1]}"
        )
    if few_table.model_name != dev_table.model_name:
        raise ValueError(
            f"Embedding model mismatch few-shot/dev: {few_table.model_name!r} vs {dev_table.model_name!r}"
        )
    if not args.allow_query_few_overlap:
        overlap = few_table.keys & dev_table.keys
        if overlap:
            raise ValueError(
                f"Detected {len(overlap)} overlapping few-shot/dev keys; sample={sorted(list(overlap))[:5]}"
            )
    dev_mat = dev_table.matrix
    dev_mat = l2_normalize(dev_mat)
    dev_lookup = compute_topk_lookup(dev_table, dev_mat, few_table, few_mat, args.top_k)
    write_archive_jsonl(
        args.out_dir / f"dev_top{args.top_k}_fewshot_medcpt_article.jsonl",
        split_name="dev",
        query_source=args.dev_path,
        few_shot_source=args.few_shot_path,
        embedding_model=few_table.model_name,
        embedding_revision=few_table.model_revision,
        embedding_dim=int(few_table.matrix.shape[1]),
        lookup=dev_lookup,
        top_k=args.top_k,
    )
    print(f"dev archive rows={len(dev_lookup)}")

    test_table = load_embedding_jsonl(args.test_path, expected_model=args.expected_model)
    if few_table.matrix.shape[1] != test_table.matrix.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch few-shot/test: {few_table.matrix.shape[1]} vs {test_table.matrix.shape[1]}"
        )
    if few_table.model_name != test_table.model_name:
        raise ValueError(
            f"Embedding model mismatch few-shot/test: {few_table.model_name!r} vs {test_table.model_name!r}"
        )
    if not args.allow_query_few_overlap:
        overlap = few_table.keys & test_table.keys
        if overlap:
            raise ValueError(
                f"Detected {len(overlap)} overlapping few-shot/test keys; sample={sorted(list(overlap))[:5]}"
            )
    test_mat = test_table.matrix
    test_mat = l2_normalize(test_mat)
    test_lookup = compute_topk_lookup(test_table, test_mat, few_table, few_mat, args.top_k)
    write_archive_jsonl(
        args.out_dir / f"test_top{args.top_k}_fewshot_medcpt_article.jsonl",
        split_name="test",
        query_source=args.test_path,
        few_shot_source=args.few_shot_path,
        embedding_model=few_table.model_name,
        embedding_revision=few_table.model_revision,
        embedding_dim=int(few_table.matrix.shape[1]),
        lookup=test_lookup,
        top_k=args.top_k,
    )
    print(f"test archive rows={len(test_lookup)}")

    print(
        "Done. model=%s revision=%s dim=%d top_k=%d"
        % (
            few_table.model_name,
            few_table.model_revision,
            int(few_table.matrix.shape[1]),
            args.top_k,
        )
    )


if __name__ == "__main__":
    main()
