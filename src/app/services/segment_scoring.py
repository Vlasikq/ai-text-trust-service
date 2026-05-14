"""Скоринг по предложениям: разбиение текста на окна и оценка каждого окна детектором.

Переиспользуется в (а) эндпоинте `/api/v1/analyze/segments` и (б) ноутбуках.
Старый `scripts/segment_scoring.py` оставлен тонкой обёрткой для совместимости.

Допущения:
  - Функция предсказания `predict_fn(text) -> float` возвращает P(AI) одного
    окна. Сегмент-уровень — TF-IDF/каскад БЕЗ overhead на калибратор/explain
    (в endpoint собираем простой predict_fn, см. routers/analyze.py).
  - Разбиение по предложениям — эвристическое (regex по `.!?` + заглавная).
    Это явно проговорено в дисклеймере UI как ограничение POC.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class SegmentScore:
    """Один сегмент: positions в исходном тексте + предсказание."""

    start_sent: int
    end_sent: int  # exclusive
    start_char: int  # inclusive
    end_char: int  # exclusive
    text: str
    prob_ai: float


@dataclass
class Sentence:
    """Предложение с позициями в исходном тексте."""

    text: str
    start: int
    end: int


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[А-ЯA-Z])")


def split_sentences(text: str) -> list[Sentence]:
    """Разбить текст на предложения, сохранив позиции в исходной строке.

    Простая эвристика: разделение по `.!?` + пробел + заглавная.
    Ограничения (известные, документированы): кавычки, сокращения (т.е., г.),
    числа с точкой (3.14). Для POC достаточно.

    Возвращает список `Sentence` с char-ranges; позиции считаются по исходной
    `text` (без strip), чтобы фронт мог рисовать подсветку поверх оригинала.
    """
    if not text.strip():
        return []
    out: list[Sentence] = []
    text_len = len(text)
    # Найдём все split-точки и разрежем по ним.
    splits = [m.start() for m in _SENT_SPLIT_RE.finditer(text)]
    boundaries = [0, *splits, text_len]
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        # split-точка включает разделитель `.!?`, потом whitespace.
        # Реальное начало следующего предложения — после whitespace.
        if i > 0:
            # Конец предыдущего совпадает с началом whitespace; сдвинем cursor
            # на первый non-space char после split-точки.
            while start < text_len and text[start].isspace():
                start += 1
        end = boundaries[i + 1]
        # Тримим trailing whitespace для display, но позиции считаем «как есть»,
        # чтобы фронт корректно подсвечивал диапазоны.
        sent_text = text[start:end].rstrip()
        if not sent_text:
            continue
        # end в Sentence — конец significant-части (без trailing whitespace).
        out.append(Sentence(text=sent_text, start=start, end=start + len(sent_text)))
    return out


def make_windows(
    sentences: list[Sentence],
    *,
    window_size: int = 3,
    step: int = 2,
) -> list[tuple[int, int]]:
    """Сформировать индексы окон: (start_sent, end_sent), end exclusive.

    `step=2` — дефолт API-уровня (отличается от scripts/segment_scoring.py
    с дефолтом 1): меньше окон → быстрее на проде, грубее по позиции.
    Для длинных текстов снимает квадратный rate.

    Если sentences меньше window_size — одно окно на весь текст
    (чтобы короткие тексты тоже получали оценку).
    """
    if not sentences:
        return []
    n = len(sentences)
    if n <= window_size:
        return [(0, n)]
    out: list[tuple[int, int]] = []
    for i in range(0, n - window_size + 1, step):
        out.append((i, i + window_size))
    # Если последнее окно не дотягивает до конца — добавим закрывающее.
    if out and out[-1][1] < n:
        out.append((n - window_size, n))
    return out


def score_segments(
    text: str,
    predict_fn,
    *,
    window_size: int = 3,
    step: int = 2,
    max_windows: int = 30,
) -> list[SegmentScore]:
    """Разбить text → окна → predict для каждого. Возвращает упорядоченный список.

    `max_windows` — защита от слишком длинных текстов: если окон больше,
    равномерно прореживаем (сохраняя первое и последнее), чтобы не съедать
    CPU/время. На лимите 8000 символов и default step=2 это не срабатывает
    в типичном случае, но защита оставлена.
    """
    sentences = split_sentences(text)
    windows = make_windows(sentences, window_size=window_size, step=step)
    if len(windows) > max_windows:
        stride = max(1, len(windows) // max_windows)
        windows = windows[::stride][:max_windows]

    out: list[SegmentScore] = []
    for start_idx, end_idx in windows:
        chunk = sentences[start_idx:end_idx]
        if not chunk:
            continue
        start_char = chunk[0].start
        end_char = chunk[-1].end
        window_text = text[start_char:end_char]
        prob = float(predict_fn(window_text))
        out.append(
            SegmentScore(
                start_sent=start_idx,
                end_sent=end_idx,
                start_char=start_char,
                end_char=end_char,
                text=window_text,
                prob_ai=prob,
            )
        )
    return out


def detect_transition(probs: list[float], threshold: float = 0.3) -> bool:
    """Эвристика «переход LOW↔HIGH»: соседние сегменты с разницей > threshold
    или общий разброс > 1.5×threshold. Используется в summary."""
    if not probs:
        return False
    if max(probs) - min(probs) > threshold * 1.5:
        return True
    for i in range(1, len(probs)):
        if abs(probs[i] - probs[i - 1]) > threshold:
            return True
    return False
