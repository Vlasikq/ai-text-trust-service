"""
Извлечение стилометрических фичей для детекции AI-текстов

45 topic-инвариантных признаков, сгруппированных по типу:
  - base (10): из Baseline B
  - lexical (6): разброс длин, hapax, кириллица, цифры
  - punctuation (7): точка с запятой, двоеточие, тире, скобки и т.д.
  - structure (4): длины предложений, первое/последнее предложение
  - function_words (12): discourse markers + LLM-маркеры (нормализовано на word_count)
  - ngram_entropy (3): энтропия символьных и словных n-грамм
  - repetition (2): self-repetition rate, MATTR-50

Все частотные фичи нормализованы на word_count.
function_words: single-word маркеры считаются по точному совпадению слова,
multi-word - по вхождению подстроки в текст (для "кроме того" и "в целом")
"""

from __future__ import annotations

__all__ = ["extract_all", "ALL_FEATURES", "FEATURE_GROUPS"]

import re
from collections import Counter
from math import log2

import numpy as np
import pandas as pd

FORMULAIC_RE = re.compile(
    r"в заключение|таким образом|подводя итог|в целом,?\s|резюмируя|итак,",
    re.IGNORECASE,
)

SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
WORD_RE = re.compile(r"[а-яёa-z]+", re.IGNORECASE)

# discourse markers
DISCOURSE_WORDS = ["однако", "например", "следовательно", "кроме того", "в целом"]
# LLM-маркеры (из log-odds в 02_eda)
LLM_MARKERS = [
    "является", "определённый", "контекст", "аспект",
    "значительный", "существенный", "ключевой",
]

# top-10 русских стоп-слов (для полноты, но скорее шум)

def _split_sentences(text: str) -> list[str]:
    parts = SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


def _word_list(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def _char_ngrams(text: str, n: int) -> list[str]:
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum(
        (c / total) * log2(c / total) for c in counts.values() if c > 0
    )


def _mattr(words: list[str], window: int = 50) -> float:
    """Moving Average Type-Token Ratio с окном 50 слов"""
    if len(words) < window:
        return len(set(words)) / max(len(words), 1)
    ttrs = []
    for i in range(len(words) - window + 1):
        chunk = words[i:i + window]
        ttrs.append(len(set(chunk)) / window)
    return float(np.mean(ttrs))


def _self_repetition_rate(words: list[str], n: int = 3) -> float:
    """Доля повторяющихся word n-грамм"""
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / max(len(ngrams), 1)


def _count_normalized(text_lower: str, words: list[str], pattern: str) -> float:
    """Частота паттерна, нормализованная на кол-во слов

    Multi-word (пробел в pattern): ищет подстроку в тексте
    Single-word: точное совпадение по списку слов
    """
    wc = max(len(words), 1)
    if " " in pattern:
        return text_lower.count(pattern) / wc
    return sum(1 for w in words if w == pattern) / wc


# main function

FEATURE_GROUPS = {
    "base": [
        "text_len", "word_count", "sentence_count", "avg_word_len",
        "avg_sent_len", "comma_density", "question_marks", "exclamation_marks",
        "ttr", "ends_formulaic",
    ],
    "lexical": [
        "std_word_len", "max_word_len", "hapax_ratio",
        "long_word_ratio", "short_word_ratio", "cyrillic_ratio", "digit_ratio",
    ],
    "punctuation": [
        "semicolons", "colons", "dashes", "parens",
        "quotes_guillemet", "ellipsis", "punct_density",
    ],
    "structure": [
        "sentence_len_std", "first_sentence_len", "last_sentence_len",
        "max_sent_len",
    ],
    "function_words": [
        "fw_однако", "fw_например", "fw_следовательно",
        "fw_кроме_того", "fw_в_целом",
        "fw_является", "fw_определённый", "fw_контекст", "fw_аспект",
        "fw_значительный", "fw_существенный", "fw_ключевой",
    ],
    "ngram_entropy": [
        "char_bigram_entropy", "char_trigram_entropy", "word_bigram_entropy",
    ],
    "repetition": [
        "self_repetition_rate", "mattr_50",
    ],
}

ALL_FEATURES = []
for group_features in FEATURE_GROUPS.values():
    ALL_FEATURES.extend(group_features)


def _extract_single(text: str) -> dict:
    """Извлекает все фичи из одного текста"""
    words = _word_list(text)
    word_lengths = [len(w) for w in words]
    sentences = _split_sentences(text)
    sent_word_counts = [len(s.split()) for s in sentences]
    wc = max(len(words), 1)
    sc = max(len(sentences), 1)
    text_lower = text.lower()

    f = {}

    # base (10)
    f["text_len"] = len(text)
    f["word_count"] = len(words)
    f["sentence_count"] = len(sentences)
    f["avg_word_len"] = float(np.mean(word_lengths)) if word_lengths else 0.0
    f["avg_sent_len"] = wc / sc
    f["comma_density"] = text.count(",") / sc
    f["question_marks"] = text.count("?") / wc
    f["exclamation_marks"] = text.count("!") / wc
    f["ttr"] = len(set(words)) / wc
    f["ends_formulaic"] = (
        int(bool(FORMULAIC_RE.search(text[-300:]))) if len(text) > 100 else 0
    )

    # lexical (7)
    f["std_word_len"] = float(np.std(word_lengths)) if word_lengths else 0.0
    f["max_word_len"] = max(word_lengths) if word_lengths else 0
    f["hapax_ratio"] = (
        sum(1 for c in Counter(words).values() if c == 1) / wc
    )
    f["long_word_ratio"] = sum(1 for w in word_lengths if w > 10) / wc
    f["short_word_ratio"] = sum(1 for w in word_lengths if w < 3) / wc
    cyr_chars = len(re.findall(r"[а-яёА-ЯЁ]", text))
    f["cyrillic_ratio"] = cyr_chars / max(len(text), 1)
    f["digit_ratio"] = len(re.findall(r"\d", text)) / max(len(text), 1)

    # punctuation (7)
    f["semicolons"] = text.count(";") / wc
    f["colons"] = text.count(":") / wc
    f["dashes"] = text.count("—") / wc  # em-dash
    f["parens"] = len(re.findall(r"[()]", text)) / wc
    f["quotes_guillemet"] = len(re.findall(r"[«»]", text)) / wc
    f["ellipsis"] = text.count("...") / wc
    f["punct_density"] = len(re.findall(r"[^\w\s]", text)) / max(len(text), 1)

    # structure (4)
    f["sentence_len_std"] = (
        float(np.std(sent_word_counts)) if len(sent_word_counts) > 1 else 0.0
    )
    f["first_sentence_len"] = len(sentences[0].split()) if sentences else 0
    f["last_sentence_len"] = len(sentences[-1].split()) if sentences else 0
    f["max_sent_len"] = max(sent_word_counts) if sent_word_counts else 0

    # function_words (12) - нормализовано на word_count
    for marker in DISCOURSE_WORDS:
        key = f"fw_{marker.replace(' ', '_')}"
        f[key] = _count_normalized(text_lower, words, marker)
    for marker in LLM_MARKERS:
        key = f"fw_{marker}"
        f[key] = _count_normalized(text_lower, words, marker)

    # ngram_entropy (3)
    f["char_bigram_entropy"] = _entropy(Counter(_char_ngrams(text_lower, 2)))
    f["char_trigram_entropy"] = _entropy(Counter(_char_ngrams(text_lower, 3)))
    word_bigrams = [
        (words[i], words[i + 1]) for i in range(len(words) - 1)
    ]
    f["word_bigram_entropy"] = _entropy(Counter(word_bigrams))

    # repetition (2)
    f["self_repetition_rate"] = _self_repetition_rate(words, n=3)
    # mattr с окном 50 слов - компромисс для коротких текстов (P5 ~ 150 слов)
    f["mattr_50"] = _mattr(words, window=50)

    return f


def extract_all(data: pd.DataFrame | str) -> pd.DataFrame:
    """Извлекает все 45 стилометрических фичей (см. ALL_FEATURES)."""
    if isinstance(data, str):
        data = pd.DataFrame({"text": [data]})

    records = []
    for text in data["text"]:
        records.append(_extract_single(text))

    return pd.DataFrame(records, index=data.index)
