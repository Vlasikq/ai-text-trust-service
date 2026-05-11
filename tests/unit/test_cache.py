"""Tests for app.cache — DetectionCache."""

from app.cache import DetectionCache
from app.detectors.base import DetectionResult


def _make_result(prob: float = 0.85) -> DetectionResult:
    return DetectionResult(prob_ai=prob, method="tfidf", inference_ms=5.0)


class TestDetectionCache:
    def test_miss_returns_none(self):
        cache = DetectionCache()
        assert cache.get("неизвестный текст") is None

    def test_put_and_get(self):
        cache = DetectionCache()
        result = _make_result(0.9)
        cache.put("текст", result)
        assert cache.get("текст") is not None
        assert cache.get("текст").prob_ai == 0.9

    def test_different_texts_different_results(self):
        cache = DetectionCache()
        cache.put("текст A", _make_result(0.9))
        cache.put("текст B", _make_result(0.1))
        assert cache.get("текст A").prob_ai == 0.9
        assert cache.get("текст B").prob_ai == 0.1

    def test_lru_eviction(self):
        cache = DetectionCache(maxsize=2)
        cache.put("a", _make_result(0.1))
        cache.put("b", _make_result(0.2))
        cache.put("c", _make_result(0.3))  # evicts "a"
        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.get("c") is not None

    def test_access_refreshes_lru(self):
        cache = DetectionCache(maxsize=2)
        cache.put("a", _make_result(0.1))
        cache.put("b", _make_result(0.2))
        cache.get("a")  # refresh "a"
        cache.put("c", _make_result(0.3))  # should evict "b", not "a"
        assert cache.get("a") is not None
        assert cache.get("b") is None

    def test_stats(self):
        cache = DetectionCache(maxsize=10)
        cache.put("a", _make_result())
        cache.get("a")  # hit
        cache.get("b")  # miss

        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["maxsize"] == 10
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_stats_empty(self):
        cache = DetectionCache()
        stats = cache.stats()
        assert stats["hit_rate"] == 0.0
