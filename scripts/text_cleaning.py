"""
Очистка текста от markdown-разметки и нормализация whitespace

Используется в assemble_dataset.py и при inference
Порядок: normalize_whitespace → strip_markdown (сначала убираем \n,
чтобы regex для **bold** не ломался на многострочных фрагментах)

"""

import re

# Порядок замен важен: code blocks → inline code → links → images →
# headings → bold/italic → lists → hr → cleanup
_MD_PIPELINE: list[tuple[re.Pattern, str]] = [
    # fenced code blocks (``` ... ```)
    (re.compile(r"```[\s\S]*?```"), ""),
    # inline code (`...`)
    (re.compile(r"`([^`]*)`"), r"\1"),
    # images ![alt](url)
    (re.compile(r"!\[[^\]]*\]\([^)]+\)"), ""),
    # links [text](url) → text
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),
    # headings ## ... (at line start after whitespace normalization — match at word boundary)
    (re.compile(r"(?:^|\s)#{1,6}\s+"), " "),
    # bold/italic: **..**, *..*, __..__, _.._ (non-greedy, handles multi-word)
    (re.compile(r"\*{1,3}(.+?)\*{1,3}"), r"\1"),
    (re.compile(r"_{1,3}(.+?)_{1,3}"), r"\1"),
    # unordered lists: - item, * item (at start of "sentence" after space)
    (re.compile(r"(?:^|\s)[*\-]\s+"), " "),
    # ordered lists: 1. item
    (re.compile(r"(?:^|\s)\d+\.\s+"), " "),
    # horizontal rules: ---, ***, ___
    (re.compile(r"(?:^|\s)[-*_]{3,}(?:\s|$)"), " "),
    # stray ** or __ (unclosed / residual)
    (re.compile(r"\*{2,}"), ""),
    (re.compile(r"_{2,}"), ""),
]


def strip_markdown(text: str) -> str:
    """Удаление markdown-разметки из текста.

    Финальный проход убирает residual ** и __ на случай незакрытых тегов.
    """
    for pattern, repl in _MD_PIPELINE:
        text = pattern.sub(repl, text)
    return text



def normalize_whitespace(text: str) -> str:
    """Коллапс всех переносов и множественных пробелов в один пробел."""
    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()



def clean_text(text: str) -> str:
    """Полная очистка: normalize_whitespace → strip_markdown → финальный strip."""
    text = normalize_whitespace(text)
    text = strip_markdown(text)
    # финальная нормализация (strip_markdown мог создать двойные пробелы)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text



def truncate_by_sentence(text: str, max_chars: int) -> str:
    """Обрезает текст до max_chars по границе последнего предложения."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # ищем последний конец предложения
    last_end = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
    if last_end > max_chars * 0.5:
        return truncated[:last_end + 1].strip()
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.5:
        return truncated[:last_space].strip()
    return truncated.strip()
