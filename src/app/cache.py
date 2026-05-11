"""
In-memory LRU cache for detection results.

SHA-256 of normalized text → DetectionResult.
Repeat request with the same text → 0 ms inference.
"""

import hashlib
from collections import OrderedDict

from app.detectors.base import DetectionResult


def text_hash(text: str) -> str:
    """SHA-256 hash of text. Shared by cache and persist."""
    return hashlib.sha256(text.encode()).hexdigest()


class DetectionCache:
    def __init__(self, maxsize: int = 1024):
        self._maxsize = maxsize
        self._cache: OrderedDict[str, DetectionResult] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, text: str) -> DetectionResult | None:
        key = text_hash(text)
        result = self._cache.get(key)
        if result is not None:
            self._hits += 1
            self._cache.move_to_end(key)
            return result
        self._misses += 1
        return None

    def put(self, text: str, result: DetectionResult) -> None:
        key = text_hash(text)
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
