"""Пайплайн предобработки текста для сервисного слоя.

Те же clean_text и truncate_by_sentence применяются и на инференсе, и при
сборке датасета (scripts/assemble_dataset.py) — идентичность гарантируют
31 тест в tests/unit/test_text_cleaning.py.
"""

from dataclasses import dataclass, field

from app.config import Settings
from app.preprocessing.text_cleaning import clean_text, truncate_by_sentence
from app.schemas import WarningCode

_CYRILLIC_MIN_RATIO = 0.5


def _cyrillic_ratio(text: str) -> float:
    alpha = 0
    cyr = 0
    for ch in text:
        if ch.isalpha():
            alpha += 1
            if "Ѐ" <= ch <= "ӿ":
                cyr += 1
    if alpha == 0:
        return 0.0
    return cyr / alpha


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

        if _cyrillic_ratio(cleaned) < _CYRILLIC_MIN_RATIO:
            warnings.append(WarningCode.non_russian_text)

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
