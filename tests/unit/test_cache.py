"""Тесты app.cache — DetectionCache."""

from app.cache import DetectionCache
from app.detectors.base import DetectionResult

_MV = "ru-detector-0.1.0"


def _make_result(prob: float = 0.85) -> DetectionResult:
    return DetectionResult(prob_ai=prob, method="tfidf", inference_ms=5.0)


class TestDetectionCache:
    def test_miss_returns_none(self):
        cache = DetectionCache()
        assert cache.get("неизвестный текст", _MV) is None

    def test_put_and_get(self):
        cache = DetectionCache()
        result = _make_result(0.9)
        cache.put("текст", result, _MV)
        assert cache.get("текст", _MV) is not None
        assert cache.get("текст", _MV).prob_ai == 0.9

    def test_different_texts_different_results(self):
        cache = DetectionCache()
        cache.put("текст A", _make_result(0.9), _MV)
        cache.put("текст B", _make_result(0.1), _MV)
        assert cache.get("текст A", _MV).prob_ai == 0.9
        assert cache.get("текст B", _MV).prob_ai == 0.1

    def test_lru_eviction(self):
        cache = DetectionCache(maxsize=2)
        cache.put("a", _make_result(0.1), _MV)
        cache.put("b", _make_result(0.2), _MV)
        cache.put("c", _make_result(0.3), _MV)  # evicts "a"
        assert cache.get("a", _MV) is None
        assert cache.get("b", _MV) is not None
        assert cache.get("c", _MV) is not None

    def test_access_refreshes_lru(self):
        cache = DetectionCache(maxsize=2)
        cache.put("a", _make_result(0.1), _MV)
        cache.put("b", _make_result(0.2), _MV)
        cache.get("a", _MV)  # refresh "a"
        cache.put("c", _make_result(0.3), _MV)  # should evict "b", not "a"
        assert cache.get("a", _MV) is not None
        assert cache.get("b", _MV) is None

    def test_stats(self):
        cache = DetectionCache(maxsize=10)
        cache.put("a", _make_result(), _MV)
        cache.get("a", _MV)  # hit
        cache.get("b", _MV)  # miss

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

    def test_different_model_versions_are_isolated(self):
        """После reload новой версии модели старые записи не отдаются — иначе
        prob_ai от старого артефакта подаётся в калибратор и verdict свежей версии."""
        cache = DetectionCache()
        cache.put("одинаковый текст", _make_result(0.9), "v1")
        assert cache.get("одинаковый текст", "v1") is not None
        assert cache.get("одинаковый текст", "v2") is None
        cache.put("одинаковый текст", _make_result(0.1), "v2")
        # v1 запись осталась нетронутой.
        assert cache.get("одинаковый текст", "v1").prob_ai == 0.9
        assert cache.get("одинаковый текст", "v2").prob_ai == 0.1
