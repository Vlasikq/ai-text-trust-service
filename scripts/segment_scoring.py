"""Per-sentence scoring (тонкая обёртка над `app.services.segment_scoring`).

Реализация переехала в `src/app/services/segment_scoring.py` — тот же модуль
используется endpoint'ом `/api/v1/analyze/segments`. Этот файл оставлен как
stable-API для notebook'а `08_extended_analysis.ipynb` и для
`scripts/data_collection/generate_edited_ai.py` — оба импортируют
`split_sentences`, `score_segments`, `summarize_segments`.

Внимание: исторический дефолт `step=1` и формат возврата `summarize_segments`
сохранены здесь, чтобы не ломать существующих потребителей. Новый код должен
импортировать из `app.services.segment_scoring`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.services.segment_scoring import (
    SegmentScore as _ServiceSegmentScore,
)
from app.services.segment_scoring import (
    detect_transition,
)
from app.services.segment_scoring import (
    make_windows as _service_make_windows,
)
from app.services.segment_scoring import (
    score_segments as _service_score_segments,
)
from app.services.segment_scoring import (
    split_sentences as _service_split_sentences,
)


@dataclass
class SegmentScore:
    """Notebook-совместимая форма сегмента (без char-ranges)."""

    start_sent: int
    end_sent: int
    text: str
    prob_ai: float

    @classmethod
    def _from_service(cls, s: _ServiceSegmentScore) -> "SegmentScore":
        return cls(start_sent=s.start_sent, end_sent=s.end_sent, text=s.text, prob_ai=s.prob_ai)


def split_sentences(text: str) -> list[str]:
    """Plain-string список (тип, который ноутбуки ожидают)."""
    return [s.text for s in _service_split_sentences(text)]


def make_windows(
    sentences: list[str], window_size: int = 3, step: int = 1
) -> list[tuple[int, int, str]]:
    """Старый формат: (start, end, joined_text). Дефолт step=1 сохранён."""
    # Соберём «синтетические» Sentence-объекты: позиции для notebook-кода
    # не важны, нам нужны только индексы и склейка текста.
    from app.services.segment_scoring import Sentence

    cursor = 0
    fake = []
    for s in sentences:
        fake.append(Sentence(text=s, start=cursor, end=cursor + len(s)))
        cursor += len(s) + 1  # +1 за условный пробел между предложениями
    pairs = _service_make_windows(fake, window_size=window_size, step=step)
    return [(start, end, " ".join(sentences[start:end])) for start, end in pairs]


def score_segments(
    text: str,
    predict_fn,
    window_size: int = 3,
    step: int = 1,
    max_windows: int = 30,
) -> list[SegmentScore]:
    """Notebook-совместимый score_segments (default step=1)."""
    out = _service_score_segments(
        text, predict_fn, window_size=window_size, step=step, max_windows=max_windows
    )
    return [SegmentScore._from_service(s) for s in out]


def summarize_segments(scores: list[SegmentScore]) -> dict:
    """Агрегированная статистика по сегментам — формат ноутбуков."""
    if not scores:
        return {"n_segments": 0}

    probs = [s.prob_ai for s in scores]
    return {
        "n_segments": len(scores),
        "mean_prob": float(np.mean(probs)),
        "std_prob": float(np.std(probs)),
        "min_prob": float(np.min(probs)),
        "max_prob": float(np.max(probs)),
        "n_high": sum(1 for p in probs if p >= 0.7),
        "n_medium": sum(1 for p in probs if 0.3 <= p < 0.7),
        "n_low": sum(1 for p in probs if p < 0.3),
        "transition_detected": detect_transition(probs),
    }
