"""Few-shot retrieval utilities backed by MedCPT archive JSONL files."""

from __future__ import annotations

import csv
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .data_io import CANONICAL_TO_TITLE, LABEL_ORDER
from .prompts import FewShotExample

ALLOWED_K_SHOTS: Tuple[int, ...] = (0, 5, 10)

DEFAULT_FEW_SHOT_POOL_PATH = Path("data/few_shot/few_shot_50.csv")
DEFAULT_RETRIEVAL_ARCHIVES: Dict[str, Path] = {
    "dev": Path("data/embeddings/retrieval_archives/dev_top10_fewshot_medcpt_article.jsonl"),
    "test": Path("data/embeddings/retrieval_archives/test_top10_fewshot_medcpt_article.jsonl"),
}

_INTEGER_PATTERN = re.compile(r"^[+-]?\d+$")
_INTEGER_EQUIVALENT_FLOAT_PATTERN = re.compile(r"^[+-]?\d+\.0+$")


def normalize_split(split: str) -> str:
    normalized = str(split).strip().lower()
    if normalized not in DEFAULT_RETRIEVAL_ARCHIVES:
        allowed = ", ".join(sorted(DEFAULT_RETRIEVAL_ARCHIVES))
        raise ValueError(f"Unsupported retrieval split '{split}'. Allowed: {allowed}.")
    return normalized


def _as_key(dataset: str, example_id: str) -> str:
    return f"{str(dataset).strip().lower()}::{str(example_id).strip()}"


def _parse_label_value(raw_value: object) -> Optional[int]:
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        raise ValueError(f"Unsupported boolean label value '{raw_value}'.")
    if isinstance(raw_value, int):
        value = raw_value
    elif isinstance(raw_value, float):
        if not raw_value.is_integer():
            raise ValueError(f"Unsupported non-integer label value '{raw_value}'.")
        value = int(raw_value)
    else:
        text = str(raw_value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return None
        if _INTEGER_PATTERN.fullmatch(text):
            value = int(text)
        elif _INTEGER_EQUIVALENT_FLOAT_PATTERN.fullmatch(text):
            value = int(text.split(".", 1)[0])
        else:
            raise ValueError(
                f"Label value must be integer-equivalent or null-like; received '{raw_value}'."
            )

    if value not in {1, 0, -1}:
        raise ValueError(f"Unsupported label value '{raw_value}'.")
    return value


def _coerce_k_shot(k_shot: object) -> int:
    if isinstance(k_shot, bool):
        raise ValueError(f"k_shot={k_shot!r} is invalid; expected one of {ALLOWED_K_SHOTS}.")
    try:
        parsed = int(k_shot)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"k_shot={k_shot!r} is invalid; expected one of {ALLOWED_K_SHOTS}.") from exc

    if parsed not in ALLOWED_K_SHOTS:
        allowed = ", ".join(str(value) for value in ALLOWED_K_SHOTS)
        raise ValueError(f"k_shot={parsed} is unsupported for retrieval archives. Allowed: {allowed}.")
    return parsed


def _match_rank(match: object, *, path: Path, line_no: int) -> int:
    if not isinstance(match, dict):
        raise ValueError(f"Archive row {line_no} in {path} has non-object match entry: {match!r}")
    try:
        return int(match.get("rank", 10_000_000))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Archive row {line_no} in {path} has invalid rank value: {match.get('rank')!r}"
        ) from exc


@lru_cache(maxsize=4)
def load_few_shot_pool(path_str: str) -> Dict[str, FewShotExample]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Few-shot pool not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"dataset", "id", "report", *LABEL_ORDER}
        missing = [column for column in required_columns if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Few-shot pool missing required columns {missing}: {path}")

        pool: Dict[str, FewShotExample] = {}
        for row in reader:
            line_no = reader.line_num
            dataset = str(row.get("dataset", "")).strip().lower()
            example_id = str(row.get("id", "")).strip()
            report_text = str(row.get("report", "")).strip()
            if not dataset or not example_id or not report_text:
                continue

            labels_title_case: Dict[str, Optional[int]] = {}
            for canonical_label in LABEL_ORDER:
                title_label = CANONICAL_TO_TITLE[canonical_label]
                raw_label_value = row.get(canonical_label)
                try:
                    labels_title_case[title_label] = _parse_label_value(raw_label_value)
                except ValueError as exc:
                    raise ValueError(
                        f"Few-shot pool row {line_no} in {path} has invalid label "
                        f"for '{canonical_label}': {raw_label_value!r}"
                    ) from exc

            pool[_as_key(dataset, example_id)] = (report_text, labels_title_case)

    return pool


@lru_cache(maxsize=4)
def load_retrieval_archive(path_str: str) -> Dict[str, List[str]]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Retrieval archive not found: {path}")

    archive: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in retrieval archive {path} line {line_no}: {exc.msg}"
                ) from exc

            if not isinstance(payload, dict):
                raise ValueError(
                    f"Archive row {line_no} in {path} must be a JSON object, got {type(payload).__name__}."
                )

            query_dataset = str(payload.get("query_dataset", "")).strip().lower()
            query_id = str(payload.get("query_id", "")).strip()
            if not query_dataset or not query_id:
                raise ValueError(f"Archive row {line_no} in {path} missing query_dataset/query_id")

            matches = payload.get("matches", [])
            if not isinstance(matches, list):
                raise ValueError(f"Archive row {line_no} in {path} has non-list matches")

            ordered = sorted(matches, key=lambda item: _match_rank(item, path=path, line_no=line_no))
            match_keys: List[str] = []
            for match in ordered:
                if not isinstance(match, dict):
                    raise ValueError(
                        f"Archive row {line_no} in {path} has non-object match entry: {match!r}"
                    )
                few_dataset = str(match.get("few_shot_dataset", "")).strip().lower()
                few_id = str(match.get("few_shot_id", "")).strip()
                if not few_dataset or not few_id:
                    raise ValueError(
                        f"Archive row {line_no} in {path} has match missing few_shot_dataset/few_shot_id"
                    )
                match_keys.append(_as_key(few_dataset, few_id))

            archive[_as_key(query_dataset, query_id)] = match_keys

    return archive


class RetrievalFewShotResolver:
    def __init__(
        self,
        *,
        split: str,
        archive_path: Optional[Path] = None,
        pool_path: Optional[Path] = None,
        strict: bool = True,
    ) -> None:
        self.split = normalize_split(split)
        self.archive_path = Path(archive_path) if archive_path is not None else DEFAULT_RETRIEVAL_ARCHIVES[self.split]
        self.pool_path = Path(pool_path) if pool_path is not None else DEFAULT_FEW_SHOT_POOL_PATH
        self.strict = bool(strict)

        self._archive = load_retrieval_archive(str(self.archive_path))
        self._pool = load_few_shot_pool(str(self.pool_path))

    def resolve(self, query_dataset: str, query_id: str, k_shot: int) -> List[FewShotExample]:
        requested = _coerce_k_shot(k_shot)
        if requested == 0:
            return []

        key = _as_key(query_dataset, query_id)
        match_keys = self._archive.get(key)
        if not match_keys:
            if self.strict:
                raise KeyError(
                    f"No retrieval matches found for query '{key}' in archive {self.archive_path}."
                )
            return []

        if self.strict and len(match_keys) < requested:
            raise ValueError(
                f"Archive {self.archive_path} has only {len(match_keys)} matches for '{key}', "
                f"but k_shot={requested} was requested."
            )

        selected = match_keys[:requested]
        missing = [candidate for candidate in selected if candidate not in self._pool]
        if missing and self.strict:
            raise KeyError(
                f"Missing few-shot rows in pool {self.pool_path} for candidates: {missing[:5]}"
            )

        resolved: List[FewShotExample] = []
        for candidate in selected:
            example = self._pool.get(candidate)
            if example is not None:
                resolved.append(example)
        return resolved


__all__ = [
    "ALLOWED_K_SHOTS",
    "DEFAULT_FEW_SHOT_POOL_PATH",
    "DEFAULT_RETRIEVAL_ARCHIVES",
    "RetrievalFewShotResolver",
    "normalize_split",
]

