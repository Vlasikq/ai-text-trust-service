"""
In-memory LRU cache for detection results.

Ключ — `{model_version}:{sha256(text)}`. При смене артефакта (новый model_version)
старые записи не отдаются: иначе после reload модель v2 возвращала бы prob_ai,
посчитанный моделью v1, и калибровка с verdict разъехались бы со свежим артефактом.
В БД хранится «чистый» `text_hash(text)` — там дедупликация по тексту, не по версии.
"""

import hashlib
from collections import OrderedDict

from app.detectors.base import DetectionResult


def text_hash(text: str) -> str:
    """SHA-256 текста. Используется для дедупликации в БД (persist)."""
    return hashlib.sha256(text.encode()).hexdigest()


def _cache_key(text: str, model_version: str) -> str:
    return f"{model_version}:{text_hash(text)}"


class DetectionCache:
    def __init__(self, maxsize: int = 1024):
        self._maxsize = maxsize
        self._cache: OrderedDict[str, DetectionResult] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, text: str, model_version: str) -> DetectionResult | None:
        key = _cache_key(text, model_version)
        result = self._cache.get(key)
        if result is not None:
            self._hits += 1
            self._cache.move_to_end(key)
            return result
        self._misses += 1
        return None

    def put(self, text: str, result: DetectionResult, model_version: str) -> None:
        key = _cache_key(text, model_version)
        self._cache[key] = result
        self._cache.move_to_end(key)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "maxsize": self._maxsize,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
        }
