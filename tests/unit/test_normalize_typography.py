"""Тесты для scripts/normalize_typography.py — замены типографии и идемпотентность.

Модуль остаётся в scripts/ потому что используется только из notebook
`10_format_confound_ablation.ipynb` (не runtime). Импортируем напрямую — pytest
держит корень проекта на sys.path через `pyproject.toml:tool.pytest.pythonpath`.
"""

from scripts.normalize_typography import count_typography_markers, normalize_typography


class TestDashes:
    def test_em_dash(self):
        assert normalize_typography("слово — слово") == "слово - слово"

    def test_en_dash(self):
        assert normalize_typography("2010–2020") == "2010-2020"

    def test_minus_sign(self):
        assert normalize_typography("температура −5") == "температура -5"

    def test_all_dash_variants_collapse(self):
        text = "a‐b ‑c ‒d –e —f ―g −h"
        assert normalize_typography(text) == "a-b -c -d -e -f -g -h"


class TestDoubleQuotes:
    def test_guillemets(self):
        assert normalize_typography("слово «цитата» конец") == 'слово "цитата" конец'

    def test_curly_quotes(self):
        assert normalize_typography("слово “цитата” конец") == 'слово "цитата" конец'

    def test_low_high_quotes(self):
        assert normalize_typography("„низ‟") == '"низ"'


class TestSingleQuotes:
    def test_curly_apostrophe(self):
        assert normalize_typography("it’s") == "it's"

    def test_left_curly_single(self):
        assert normalize_typography("‘слово’") == "'слово'"


class TestWhitespace:
    def test_nbsp_replaced(self):
        assert normalize_typography("слово слово") == "слово слово"

    def test_double_space_collapsed(self):
        assert normalize_typography("слово  слово") == "слово слово"

    def test_multiple_whitespace_collapsed(self):
        assert normalize_typography("слово\t\n  слово") == "слово слово"

    def test_trailing_whitespace_stripped(self):
        assert normalize_typography("  слово  ") == "слово"


class TestPreserved:
    """Что НЕ должно меняться: многоточие, скобки, обычная пунктуация."""

    def test_ellipsis_preserved(self):
        assert normalize_typography("слово… конец") == "слово… конец"

    def test_parentheses_preserved(self):
        assert normalize_typography("слово (вставка) конец") == "слово (вставка) конец"

    def test_regular_punctuation(self):
        assert normalize_typography("слово, ещё. вопрос?") == "слово, ещё. вопрос?"

    def test_ascii_unchanged(self):
        assert normalize_typography("plain ASCII text") == "plain ASCII text"


class TestIdempotence:
    """normalize(normalize(t)) == normalize(t) на разнородных текстах."""

    @staticmethod
    def _check(text: str):
        once = normalize_typography(text)
        twice = normalize_typography(once)
        assert once == twice, f"not idempotent on: {text!r}"

    def test_simple(self):
        self._check("слово — «цитата» конец")

    def test_mixed_typography(self):
        self._check("В 2022 г. — “Он сказал ‘да’”.")

    def test_empty(self):
        self._check("")

    def test_only_ascii(self):
        self._check("plain text 123")

    def test_only_typography(self):
        self._check("«»—– ’“”")


class TestCountMarkers:
    def test_counts_em_dash(self):
        counts = count_typography_markers("a — b — c")
        assert counts["em_dash_U2014"] == 2

    def test_counts_guillemets(self):
        counts = count_typography_markers("«один» «два»")
        assert counts["guillemet_left_U00AB"] == 2
        assert counts["guillemet_right_U00BB"] == 2

    def test_zero_for_missing(self):
        counts = count_typography_markers("plain ASCII")
        assert all(v == 0 for v in counts.values())

    def test_after_normalization_all_zero(self):
        text = "слово — «цитата» — ’конец’"
        normalized = normalize_typography(text)
        counts = count_typography_markers(normalized)
        assert all(v == 0 for v in counts.values()), counts
