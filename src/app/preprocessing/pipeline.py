"""
Text preprocessing pipeline for the service layer.

Same clean_text / truncate_by_sentence applied at inference and at training
(scripts/assemble_dataset.py) — 31 tests guarantee identity.
"""

from dataclasses import dataclass, field

from app.config import Settings
from app.preprocessing.text_cleaning import clean_text, truncate_by_sentence
from app.schemas import WarningCode


@dataclass
class PreprocessResult:
    text: str
    original_length: int
    cleaned_length: int
    was_truncated: bool = False
    warnings: list[WarningCode] = field(default_factory=list)


class TextPreprocessor:
    def __init__(self, settings: Settings):
        self._min_chars = settings.min_chars
        self._max_chars = settings.max_chars

    def __call__(self, raw_text: str) -> PreprocessResult:
        original_length = len(raw_text)

        cleaned = clean_text(raw_text)
        cleaned_length = len(cleaned)

        warnings: list[WarningCode] = []

        if cleaned_length < self._min_chars:
            warnings.append(WarningCode.text_too_short)
            return PreprocessResult(
                text=cleaned,
                original_length=original_length,
                cleaned_length=cleaned_length,
                warnings=warnings,
            )

        was_truncated = False
        if cleaned_length > self._max_chars:
            cleaned = truncate_by_sentence(cleaned, self._max_chars)
            was_truncated = True
            cleaned_length = len(cleaned)
            warnings.append(WarningCode.text_truncated)

        return PreprocessResult(
            text=cleaned,
            original_length=original_length,
            cleaned_length=cleaned_length,
            was_truncated=was_truncated,
            warnings=warnings,
        )
