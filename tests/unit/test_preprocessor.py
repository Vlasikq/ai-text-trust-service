"""Tests for app.preprocessing.pipeline — TextPreprocessor."""

from app.config import Settings
from app.preprocessing.pipeline import TextPreprocessor
from app.schemas import WarningCode


def _make_preprocessor(**overrides) -> TextPreprocessor:
    return TextPreprocessor(Settings(**overrides))


class TestNormalText:
    def test_clean_text_applied(self):
        pp = _make_preprocessor(min_chars=5)
        result = pp("  Привет   мир  ")
        assert result.text == "Привет мир"

    def test_markdown_stripped(self):
        pp = _make_preprocessor(min_chars=5)
        result = pp("## Заголовок **жирный**")
        assert "#" not in result.text
        assert "**" not in result.text

    def test_no_warnings_for_normal_text(self):
        pp = _make_preprocessor(min_chars=10)
        result = pp("Нормальный текст для проверки.")
        assert result.warnings == []

    def test_lengths_tracked(self):
        pp = _make_preprocessor(min_chars=5)
        raw = "  Привет   мир  "
        result = pp(raw)
        assert result.original_length == len(raw)
        assert result.cleaned_length == len("Привет мир")


class TestTooShort:
    def test_short_text_warning(self):
        pp = _make_preprocessor(min_chars=300)
        result = pp("Короткий текст.")
        assert WarningCode.text_too_short in result.warnings

    def test_short_text_not_truncated(self):
        pp = _make_preprocessor(min_chars=300)
        result = pp("Короткий текст.")
        assert result.was_truncated is False

    def test_exact_min_chars_no_warning(self):
        pp = _make_preprocessor(min_chars=10)
        result = pp("0123456789")
        assert WarningCode.text_too_short not in result.warnings


class TestTruncation:
    def test_long_text_truncated(self):
        pp = _make_preprocessor(min_chars=5, max_chars=50)
        long_text = "Первое предложение. Второе предложение. Третье предложение. Четвёртое. Пятое."
        result = pp(long_text)
        assert result.was_truncated is True
        assert WarningCode.text_truncated in result.warnings
        assert result.cleaned_length <= 50

    def test_exact_max_chars_no_truncation(self):
        pp = _make_preprocessor(min_chars=5, max_chars=100)
        text = "x" * 100
        result = pp(text)
        assert result.was_truncated is False
        assert WarningCode.text_truncated not in result.warnings


class TestEdgeCases:
    def test_empty_after_cleaning(self):
        pp = _make_preprocessor(min_chars=300)
        result = pp("   ")
        assert WarningCode.text_too_short in result.warnings
        assert result.text == ""

    def test_unicode_preserved(self):
        pp = _make_preprocessor(min_chars=5)
        result = pp("Ёжик в тумане 🦔")
        assert "Ёжик" in result.text

    def test_code_blocks_removed(self):
        pp = _make_preprocessor(min_chars=5)
        result = pp("Текст ```python\nprint('hi')\n``` конец")
        assert "print" not in result.text
        assert "Текст" in result.text
