from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

TITLE_TO_CANONICAL: Dict[str, str] = {
    "No Finding": "no_finding",
    "Enlarged Cardiomediastinum": "enlarged_cardiomediastinum",
    "Cardiomegaly": "cardiomegaly",
    "Lung Opacity": "lung_opacity",
    "Lung Lesion": "lung_lesion",
    "Edema": "edema",
    "Consolidation": "consolidation",
    "Pneumonia": "pneumonia",
    "Atelectasis": "atelectasis",
    "Pneumothorax": "pneumothorax",
    "Pleural Effusion": "pleural_effusion",
    "Pleural Other": "pleural_other",
    "Fracture": "fracture",
    "Support Devices": "support_devices",
}

LABEL_ORDER: List[str] = list(TITLE_TO_CANONICAL.values())

CANONICAL_TO_TITLE: Dict[str, str] = {value: key for key, value in TITLE_TO_CANONICAL.items()}

DATASET_CONTEXTS = {"impression", "both", "full"}
SPLIT_CONTEXTS = {"full"}

DATA_ROOT = Path("data")
GPT_MAJORITY_ROOT = DATA_ROOT / "gpt_majority_labels"

DATASET_SPECS: Dict[str, Dict[str, Path]] = {
    "mimic": {
        "labels": GPT_MAJORITY_ROOT / "mimic_gpt_majority.csv",
        "sections": DATA_ROOT / "mimic" / "labels-mimic-sections.csv",
    },
    "indiana": {
        "labels": GPT_MAJORITY_ROOT / "indiana_gpt_majority.csv",
        "sections": DATA_ROOT / "indiana" / "labels-indiana-sections.csv",
    },
    "rexgradient": {
        "labels": GPT_MAJORITY_ROOT / "rexgradient_gpt_nano_majority_20260209_141952.csv",
        "sections": DATA_ROOT / "rexgradient" / "labels-rexgradient-sections.csv",
    },
}

SPLIT_DATASET_PATHS: Dict[str, Path] = {
    "dev_all": DATA_ROOT / "dev" / "dev_all.csv",
    "test_all": DATA_ROOT / "test" / "test_all.csv",
    "indiana_dev": DATA_ROOT / "dev" / "indiana_dev.csv",
    "indiana_test": DATA_ROOT / "test" / "indiana_test.csv",
    "mimic_dev": DATA_ROOT / "dev" / "mimic_dev.csv",
    "mimic_test": DATA_ROOT / "test" / "mimic_test.csv",
    "rexgradient_dev": DATA_ROOT / "dev" / "rexgradient_dev.csv",
    "rexgradient_test": DATA_ROOT / "test" / "rexgradient_test.csv",
}

SECTION_COLUMNS: List[str] = [
    "section_impression",
    "section_findings",
    "section_findings_plus_section_impression",
    "report",
]


@dataclass(frozen=True)
class DatasetExample:
    example_id: str
    source_dataset: str
    text: str
    labels_four_class: Dict[str, Optional[int]]
    labels_binary: Dict[str, int]


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    input_context: str
    examples: List[DatasetExample]

    def __iter__(self):
        return iter(self.examples)

    def __len__(self) -> int:
        return len(self.examples)


def load_dataset(name: str, input_context: str, limit: Optional[int] = None) -> DatasetBundle:
    """Load a configured dataset with the requested report text context."""
    dataset_key = _resolve_dataset_key(name)
    if dataset_key in SPLIT_DATASET_PATHS:
        return _load_split_csv_dataset(dataset_key, input_context, limit=limit)
    return _load_section_backed_dataset(dataset_key, input_context, limit=limit)


def _resolve_dataset_key(name: str) -> str:
    normalized = name.strip().lower()
    alias_map = {
        "mimic": "mimic",
        "mimic-cxr": "mimic",
        "mimic_cxr": "mimic",
        "mimic_full": "mimic",
        "mimic-full": "mimic",
        "mimic_full_eval": "mimic",
        "mimic-full-eval": "mimic",
        "indiana": "indiana",
        "rexgradient": "rexgradient",
        "rex-gradient": "rexgradient",
        "rex_gradient": "rexgradient",
        "dev": "dev_all",
        "all_dev": "dev_all",
        "dev_all": "dev_all",
        "test": "test_all",
        "all_test": "test_all",
        "test_all": "test_all",
        "indiana_dev": "indiana_dev",
        "indiana_test": "indiana_test",
        "mimic_dev": "mimic_dev",
        "mimic_test": "mimic_test",
        "rexgradient_dev": "rexgradient_dev",
        "rexgradient_test": "rexgradient_test",
    }
    try:
        return alias_map[normalized]
    except KeyError as exc:
        expected = ", ".join(sorted({*DATASET_SPECS, *SPLIT_DATASET_PATHS}))
        raise ValueError(f"Unsupported dataset '{name}'. Expected one of: {expected}.") from exc


def _load_section_backed_dataset(dataset_key: str, input_context: str, limit: Optional[int]) -> DatasetBundle:
    context = input_context.strip().lower()
    if context not in DATASET_CONTEXTS:
        allowed = ", ".join(sorted(DATASET_CONTEXTS))
        raise ValueError(
            f"Invalid input_context '{input_context}' for dataset '{dataset_key}'. Allowed: {allowed}."
        )

    spec = DATASET_SPECS[dataset_key]
    labels_df = _read_csv(spec["labels"])
    sections_df = _read_csv(spec["sections"])

    missing_section_cols = [column for column in SECTION_COLUMNS if column not in sections_df.columns]
    if missing_section_cols:
        raise ValueError(
            f"Section file for '{dataset_key}' is missing required columns: {missing_section_cols}"
        )

    # Labels come from majority-vote CSVs; narrative sections come from the dedicated section files.
    labels_df = labels_df.drop(columns=[column for column in SECTION_COLUMNS if column in labels_df.columns])
    merged = labels_df.join(sections_df[SECTION_COLUMNS], how="inner", rsuffix="_sections")

    if limit is not None:
        merged = merged.head(limit)

    examples = [
        _build_example(
            row=row,
            dataset_name=dataset_key,
            text_selector=lambda r: _select_text_from_sections(r, context),
        )
        for _, row in merged.iterrows()
    ]

    return DatasetBundle(name=dataset_key, input_context=context, examples=examples)


def _load_split_csv_dataset(dataset_key: str, input_context: str, limit: Optional[int]) -> DatasetBundle:
    context = input_context.strip().lower()
    if context not in SPLIT_CONTEXTS:
        allowed = ", ".join(sorted(SPLIT_CONTEXTS))
        raise ValueError(
            f"Invalid input_context '{input_context}' for split dataset '{dataset_key}'. Allowed: {allowed}."
        )

    path = SPLIT_DATASET_PATHS[dataset_key]
    df = _read_csv(path)

    required_columns = ["dataset", "id", "report", *LABEL_ORDER]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Split dataset '{dataset_key}' missing required columns: {missing}")

    if limit is not None:
        df = df.head(limit)

    default_source_dataset: Optional[str] = None
    if dataset_key.endswith("_dev") or dataset_key.endswith("_test"):
        default_source_dataset = dataset_key.rsplit("_", 1)[0]

    examples = [
        _build_split_example(
            row=row,
            default_dataset=default_source_dataset,
        )
        for _, row in df.iterrows()
    ]
    return DatasetBundle(name=dataset_key, input_context=context, examples=examples)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "id" not in df.columns:
        if "example_id" in df.columns:
            df = df.rename(columns={"example_id": "id"})
        else:
            raise ValueError(f"Required id column not found in file: {path}")
    df = df.set_index("id", drop=False)
    return df


def _build_example(row: pd.Series, dataset_name: str, text_selector) -> DatasetExample:
    example_id = str(row["id"])
    text = text_selector(row)
    if not text:
        text = _normalize_text(row.get("report"))
    if not text:
        raise ValueError(f"Empty report text for example '{example_id}' in dataset '{dataset_name}'.")

    labels_four = _extract_labels(row, dataset_name, text)
    labels_binary = _collapse_to_binary(labels_four)

    return DatasetExample(
        example_id=example_id,
        source_dataset=dataset_name,
        text=text,
        labels_four_class=labels_four,
        labels_binary=labels_binary,
    )


def _build_split_example(row: pd.Series, default_dataset: Optional[str]) -> DatasetExample:
    example_id = str(row["id"])
    source_dataset_raw = _normalize_text(row.get("dataset"))
    if not source_dataset_raw:
        if not default_dataset:
            raise ValueError(
                f"Split row '{example_id}' is missing dataset and no default source dataset is available."
            )
        source_dataset_raw = default_dataset

    source_dataset = _resolve_dataset_key(source_dataset_raw)
    if source_dataset in SPLIT_DATASET_PATHS:
        if not default_dataset:
            raise ValueError(
                f"Split row '{example_id}' resolved to split dataset '{source_dataset}' "
                "instead of a base source dataset."
            )
        source_dataset = default_dataset

    text = _normalize_text(row.get("report"))
    if not text:
        raise ValueError(
            f"Empty report text for split example '{example_id}' in source dataset '{source_dataset}'."
        )

    labels_four = _extract_labels(row, source_dataset, text)
    labels_binary = _collapse_to_binary(labels_four)
    return DatasetExample(
        example_id=example_id,
        source_dataset=source_dataset,
        text=text,
        labels_four_class=labels_four,
        labels_binary=labels_binary,
    )


def _select_text_from_sections(row: pd.Series, context: str) -> str:
    impression = _normalize_text(row.get("section_impression"))
    findings = _normalize_text(row.get("section_findings"))
    combined = _normalize_text(row.get("section_findings_plus_section_impression"))
    full = _normalize_text(row.get("report"))

    if context == "impression":
        return _prefer_text(impression, combined, full)
    if context == "both":
        text = _join_sections(impression, findings)
        return text or _prefer_text(combined, full)
    if context == "full":
        return _prefer_text(full, combined, impression, findings)
    raise AssertionError("Unhandled dataset context.")


def _normalize_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _join_sections(*sections: Optional[str]) -> str:
    parts = [str(section).strip() for section in sections if section and str(section).strip()]
    return "\n\n".join(parts)


def _prefer_text(*candidates: Optional[str]) -> str:
    for text in candidates:
        normalized = _normalize_text(text)
        if normalized:
            return normalized
    return ""


def _extract_labels(row: pd.Series, dataset_name: str, _report_text: str) -> Dict[str, Optional[int]]:
    dataset = dataset_name.lower()
    raw_values: Dict[str, Optional[int]] = {}
    provided_no_finding: Optional[int] = None
    for key in LABEL_ORDER:
        raw_value = row.get(key)
        if raw_value in {"", None}:
            raw_value = row.get(CANONICAL_TO_TITLE[key])
        parsed = _parse_label_value(raw_value, dataset)
        if key == "no_finding":
            provided_no_finding = parsed
        else:
            raw_values[key] = parsed

    no_finding_value = _normalize_no_finding_value(provided_no_finding)

    ordered = {"no_finding": no_finding_value}
    ordered.update(raw_values)
    return ordered


def _parse_label_value(value: Optional[str], dataset: str) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or _is_nan(text):
        return None
    try:
        parsed = int(float(text))
    except ValueError as exc:
        raise ValueError(f"Cannot parse label value '{value}' for dataset '{dataset}'.") from exc

    if dataset == "chexpert":
        if parsed not in {0, 1}:
            raise ValueError(f"CheXpert labels must be 0 or 1, got '{parsed}'.")
    elif dataset in {"mimic", "mimic_full", "mimic_full_eval", "indiana", "rexgradient"}:
        if parsed not in {1, 0, -1}:
            raise ValueError(f"MIMIC-style labels must be one of {{1, 0, -1}}, got '{parsed}'.")
    else:
        raise ValueError(f"Unknown dataset '{dataset}' while parsing labels.")
    return parsed


def _normalize_no_finding_value(value: Optional[int]) -> Optional[int]:
    return value


def _collapse_to_binary(labels: Dict[str, Optional[int]]) -> Dict[str, int]:
    binary: Dict[str, int] = {}
    for key in LABEL_ORDER:
        value = labels.get(key)
        if key == "no_finding":
            binary[key] = 1 if value == 1 else 0
            continue
        if value is None or value == 0:
            binary[key] = 0
        elif value in {1, -1}:
            binary[key] = 1
        else:
            raise ValueError(f"Unsupported four-class value '{value}' for label '{key}'.")
    return binary


def _is_nan(value: str) -> bool:
    if isinstance(value, float):
        return math.isnan(value)
    return str(value).strip().lower() == "nan"


__all__ = [
    "CANONICAL_TO_TITLE",
    "DatasetBundle",
    "DatasetExample",
    "LABEL_ORDER",
    "load_dataset",
]
