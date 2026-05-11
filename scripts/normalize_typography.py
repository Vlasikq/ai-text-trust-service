"""Нормализация типографики: тире, кавычки, неразрывные пробелы в ASCII.

Идемпотентна: normalize_typography(normalize_typography(t)) == normalize_typography(t).
"""

import re

# Все варианты тире, дефиса и минуса → ASCII hyphen.
# Различие между em-dash и en-dash здесь несущественно.
_DASH_TABLE = str.maketrans({
    "‐": "-",  # hyphen
    "‑": "-",  # non-breaking hyphen
    "‒": "-",  # figure dash
    "–": "-",  # en dash
    "—": "-",  # em dash
    "―": "-",  # horizontal bar
    "−": "-",  # minus sign
})

# Двойные кавычки → ASCII "
_DOUBLE_QUOTE_TABLE = str.maketrans({
    "«": '"',  # « left guillemet
    "»": '"',  # » right guillemet
    "“": '"',  # " left double curly
    "”": '"',  # " right double curly
    "„": '"',  # „ low double curly
    "‟": '"',  # ‟ high reversed double curly
})

# Одинарные кавычки → ASCII '
_SINGLE_QUOTE_TABLE = str.maketrans({
    "‘": "'",  # ' left single curly
    "’": "'",  # ' right single curly / типографский апостроф
    "‚": "'",  # ‚ low single curly
    "‛": "'",  # ‛ high reversed single curly
})

# Неразрывный пробел → обычный
_NBSP_TABLE = str.maketrans({
    " ": " ",
})

_TYPOGRAPHY_TABLE = {
    **_DASH_TABLE,
    **_DOUBLE_QUOTE_TABLE,
    **_SINGLE_QUOTE_TABLE,
    **_NBSP_TABLE,
}

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_typography(text: str) -> str:
    """Привести типографские символы к ASCII-эквивалентам.

    Все виды тире/дефиса/минуса → '-', двойные кавычки (ёлочки, curly, low/high)
    → '"', одинарные curly → "'", NBSP → ' '. В конце коллапсирует подряд
    идущие пробельные символы. Не трогает многоточие, скобки, обычную пунктуацию.
    """
    text = text.translate(_TYPOGRAPHY_TABLE)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


# Маркеры для аудита корпуса: какие типографские символы присутствуют и с какой частотой.
_MARKERS_TO_COUNT = {
    "em_dash_U2014": "—",
    "en_dash_U2013": "–",
    "hyphen_U2010": "‐",
    "nb_hyphen_U2011": "‑",
    "figure_dash_U2012": "‒",
    "horiz_bar_U2015": "―",
    "minus_U2212": "−",
    "guillemet_left_U00AB": "«",
    "guillemet_right_U00BB": "»",
    "dq_curly_left_U201C": "“",
    "dq_curly_right_U201D": "”",
    "dq_low_U201E": "„",
    "dq_high_rev_U201F": "‟",
    "sq_curly_left_U2018": "‘",
    "sq_curly_right_U2019": "’",
    "sq_low_U201A": "‚",
    "sq_high_rev_U201B": "‛",
    "nbsp_U00A0": " ",
}


def count_typography_markers(text: str) -> dict[str, int]:
    """Счётчик типографских маркеров в тексте — для аудита корпуса."""
    return {name: text.count(char) for name, char in _MARKERS_TO_COUNT.items()}
