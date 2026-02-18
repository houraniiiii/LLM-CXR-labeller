"""
Prompt construction utilities for the retrieval inference pipeline.

System prompts are sourced from `system_prompts_and_lexicon.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CHEXPERT_LABELS: List[str] = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]

CHEXPERT_LABEL_TO_CANONICAL: Dict[str, str] = {
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
    "No Finding": "no_finding",
}

CHEXPERT_CANONICAL_TO_LABEL: Dict[str, str] = {
    value: key for key, value in CHEXPERT_LABEL_TO_CANONICAL.items()
}

FewShotExample = Tuple[str, Dict[str, Any]]


def title_to_canonical(label: str) -> str:
    """Translate a Title Case CheXpert label to canonical snake_case."""
    try:
        return CHEXPERT_LABEL_TO_CANONICAL[label]
    except KeyError as exc:
        normalized = label.strip()
        if normalized in CHEXPERT_LABEL_TO_CANONICAL:
            return CHEXPERT_LABEL_TO_CANONICAL[normalized]
        raise KeyError(f"Unknown CheXpert label '{label}'.") from exc


def canonical_to_title(label: str) -> str:
    """Translate a canonical snake_case label to Title Case."""
    try:
        return CHEXPERT_CANONICAL_TO_LABEL[label]
    except KeyError as exc:
        normalized = label.strip()
        if normalized in CHEXPERT_CANONICAL_TO_LABEL:
            return CHEXPERT_CANONICAL_TO_LABEL[normalized]
        raise KeyError(f"Unknown canonical CheXpert label '{label}'.") from exc


class PromptVariant(str, Enum):
    BASIC = "basic"
    CLINICAL_STEPWISE = "clinical_stepwise"
    CLINICAL_COMPACT = "clinical_compact"
    CLINICAL_STANDARD = "clinical_standard"

    @classmethod
    def from_value(cls, value: str | PromptVariant) -> PromptVariant:
        if isinstance(value, cls):
            return value
        lowered = value.lower()
        try:
            return cls(lowered)
        except ValueError as exc:
            valid = ", ".join(member.value for member in cls)
            raise ValueError(f"Unknown prompt variant '{value}'. Expected one of: {valid}.") from exc


@dataclass(frozen=True)
class PromptTemplate:
    variant: PromptVariant
    base_prompt: str = ""
    sections: Tuple[str, ...] = ()

    def render(self) -> str:
        base = self.base_prompt.strip() if self.base_prompt else ""
        section_blocks = [section.strip() for section in self.sections if section.strip()]
        if not section_blocks:
            return base
        appendix = "\n".join(section_blocks)
        return "\n\n".join([base, appendix]) if base else appendix


PROMPT_MARKDOWN_PATH = Path("system_prompts_and_lexicon.md")
_MD_PROMPTS: Optional[Dict[str, str]] = None
_MD_LEXICON: Optional[str] = None


def _load_prompt_markdown(md_path: Path) -> Tuple[Dict[str, str], Optional[str]]:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    prompts: Dict[str, List[str]] = {}
    lexicon_lines: List[str] = []
    current_prompt: Optional[str] = None
    in_lexicon = False

    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            lowered = title.lower()
            if lowered.startswith("prompt:"):
                current_prompt = title.split(":", 1)[1].strip()
                prompts[current_prompt] = []
                in_lexicon = False
                continue
            if "lexicon" in lowered:
                current_prompt = None
                in_lexicon = True
                continue
            current_prompt = None
            in_lexicon = False

        if in_lexicon:
            lexicon_lines.append(line)
        elif current_prompt is not None:
            prompts[current_prompt].append(line)

    base_prompts = {
        name: "\n".join(content).strip()
        for name, content in prompts.items()
        if any(line.strip() for line in content)
    }
    lexicon_text = "\n".join(lexicon_lines).strip() if lexicon_lines else None
    return base_prompts, lexicon_text


def _ensure_md_loaded() -> None:
    global _MD_PROMPTS, _MD_LEXICON
    if _MD_PROMPTS is not None:
        return

    if not PROMPT_MARKDOWN_PATH.exists():
        raise FileNotFoundError(f"Prompt markdown not found: {PROMPT_MARKDOWN_PATH}")

    prompts, lexicon_text = _load_prompt_markdown(PROMPT_MARKDOWN_PATH)
    if not prompts:
        raise ValueError(f"No prompt sections found in {PROMPT_MARKDOWN_PATH}")
    if not lexicon_text:
        raise ValueError(f"No lexicon section found in {PROMPT_MARKDOWN_PATH}")

    _MD_PROMPTS = prompts
    _MD_LEXICON = lexicon_text


def _get_md_prompt(name: str) -> str:
    _ensure_md_loaded()
    assert _MD_PROMPTS is not None
    try:
        return _MD_PROMPTS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_MD_PROMPTS))
        raise KeyError(f"Unknown prompt '{name}'. Available: {known}") from exc


def build_lexicon_sections() -> List[str]:
    _ensure_md_loaded()
    assert _MD_LEXICON is not None
    return [_MD_LEXICON]


# Maps supported variants to markdown prompt keys. Lexicon is always appended.
_VARIANT_SPECS: Dict[PromptVariant, str] = {
    PromptVariant.BASIC: "basic",
    PromptVariant.CLINICAL_STEPWISE: "clinical_stepwise",
    PromptVariant.CLINICAL_COMPACT: "clinical_compact",
    PromptVariant.CLINICAL_STANDARD: "clinical_standard",
}


def _compose_prompt_text(prompt_key: str) -> str:
    return PromptTemplate(
        variant=PromptVariant.CLINICAL_STANDARD,
        base_prompt=_get_md_prompt(prompt_key),
        sections=tuple(build_lexicon_sections()),
    ).render()


def build_prompt_template(variant: PromptVariant) -> PromptTemplate:
    try:
        prompt_key = _VARIANT_SPECS[variant]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unhandled prompt variant: {variant!r}") from exc

    return PromptTemplate(
        variant=variant,
        base_prompt=_get_md_prompt(prompt_key),
        sections=tuple(build_lexicon_sections()),
    )


PROMPT_TEMPLATES: Dict[PromptVariant, PromptTemplate] = {
    variant: build_prompt_template(variant) for variant in PromptVariant
}

SYSTEM_PROMPT = PROMPT_TEMPLATES[PromptVariant.CLINICAL_STANDARD].render()


class PromptBuilder:
    """Assemble chat messages for the inference runtime."""

    def __init__(
        self,
        *,
        prompt_variant: PromptVariant | str = PromptVariant.CLINICAL_STANDARD,
        system_prompt: Optional[str] = None,
        few_shots: Optional[List[FewShotExample]] = None,
    ) -> None:
        resolved_variant: Optional[PromptVariant]
        resolved_template: Optional[PromptTemplate]

        if system_prompt is not None:
            self.system_prompt = system_prompt.strip()
            resolved_variant = None
            resolved_template = None
            variant_label = "custom"
        else:
            resolved_variant = None
            resolved_template = None
            custom_key: Optional[str] = None

            if isinstance(prompt_variant, PromptVariant):
                resolved_variant = prompt_variant
            else:
                try:
                    resolved_variant = PromptVariant.from_value(prompt_variant)
                except ValueError:
                    custom_key = str(prompt_variant)

            if resolved_variant is not None:
                resolved_template = PROMPT_TEMPLATES[resolved_variant]
                self.system_prompt = resolved_template.render()
                variant_label = resolved_variant.value
            else:
                prompt_key = custom_key or "clinical_standard"
                self.system_prompt = _compose_prompt_text(prompt_key)
                variant_label = prompt_key

        self.prompt_variant: Optional[PromptVariant] = resolved_variant
        self.prompt_template: Optional[PromptTemplate] = resolved_template
        self.prompt_variant_label: str = variant_label

        self._default_few_shots: List[Tuple[str, Dict[str, Optional[int]]]] = (
            self._normalize_few_shots(few_shots) if few_shots else []
        )
        self.metadata: Dict[str, str] = {
            "few_shot_examples": str(len(self._default_few_shots)),
            "prompt_variant": variant_label,
        }

    def build_messages(
        self,
        *,
        report_text: str,
        report_id: Optional[str] = None,
        few_shots: Optional[List[FewShotExample]] = None,
    ) -> List[Dict[str, str]]:
        """Create a chat message list ready for model APIs."""
        if not isinstance(report_text, str) or not report_text.strip():
            raise ValueError("report_text must be a non-empty string.")

        payload = {
            "report_id": report_id,
            "report_text": report_text.strip(),
        }

        messages: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]

        normalized_shots: List[Tuple[str, Dict[str, Optional[int]]]] = (
            self._normalize_few_shots(few_shots)
            if few_shots is not None
            else list(self._default_few_shots)
        )
        if normalized_shots:
            self.metadata["few_shot_examples"] = str(len(normalized_shots))
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Here are reference report-label examples curated by expert radiologists that demonstrate the expected output format. "
                        "Use them as guidance before analysing the final request independently."
                    ),
                }
            )
            for example_report, example_labels in normalized_shots:
                user_payload = {
                    "report_id": None,
                    "report_text": example_report,
                }
                assistant_payload: Dict[str, Any] = {
                    "report_id": None,
                    "labels": example_labels,
                }

                messages.append({"role": "user", "content": self._serialize_payload(user_payload)})
                messages.append(
                    {
                        "role": "assistant",
                        "content": self._serialize_payload(assistant_payload),
                    }
                )
        else:
            self.metadata["few_shot_examples"] = "0"

        messages.append({"role": "user", "content": self._serialize_payload(payload)})
        return messages

    @staticmethod
    def _serialize_payload(payload: Dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _normalize_few_shots(
        self, shots: Optional[List[FewShotExample]]
    ) -> List[Tuple[str, Dict[str, Optional[int]]]]:
        if not shots:
            return []

        normalized: List[Tuple[str, Dict[str, Optional[int]]]] = []
        for entry in shots:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise ValueError("Each few-shot example must be a (report, labels) tuple.")
            report, labels = entry
            if not isinstance(report, str) or not report.strip():
                raise ValueError("Few-shot report must be a non-empty string.")
            if not isinstance(labels, dict):
                raise ValueError("Few-shot labels must be a dictionary.")

            normalized_labels: Dict[str, Optional[int]] = {}
            missing = [name for name in CHEXPERT_LABELS if name not in labels]
            if missing:
                raise ValueError(f"Few-shot labels missing keys: {missing}")

            for name in CHEXPERT_LABELS:
                value = labels.get(name)
                if value is None:
                    normalized_labels[name] = None
                elif value in {1, 0, -1}:
                    normalized_labels[name] = int(value)
                else:
                    raise ValueError(
                        f"Invalid value for few-shot label '{name}': {value!r} "
                        "(expected 1, 0, -1, or None)"
                    )

            normalized.append((report.strip(), normalized_labels))

        return normalized


__all__ = [
    "PromptBuilder",
    "PromptTemplate",
    "PromptVariant",
    "PROMPT_TEMPLATES",
    "build_prompt_template",
    "SYSTEM_PROMPT",
    "CHEXPERT_LABELS",
    "CHEXPERT_LABEL_TO_CANONICAL",
    "CHEXPERT_CANONICAL_TO_LABEL",
    "title_to_canonical",
    "canonical_to_title",
    "FewShotExample",
]
