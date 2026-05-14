"""Тесты детекторов: base, tfidf, transformer, cascade."""

from app.detectors.base import BaseDetector, DetectionResult
from app.detectors.cascade import CascadeDetector
from app.schemas import WarningCode

# ── Local fakes (см. tests/conftest.py для общего FakeDetector) ────────────────
# Здесь нужна другая семантика: is_ready() зависит от вызова load() —
# проверяем именно lifecycle BaseDetector. Conftest'овский FakeDetector
# всегда ready=True по дефолту, что прячет lifecycle-баги (например, если
# Cascade забудет вызвать load() на slow-детектора).


class FakeDetector(BaseDetector):
    """Lifecycle-aware fake: is_ready() становится True только после load()."""

    def __init__(self, prob_ai: float = 0.5, method: str = "fake"):
        self._prob_ai = prob_ai
        self._method = method
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def predict(self, text: str) -> DetectionResult:
        return DetectionResult(
            prob_ai=self._prob_ai,
            method=self._method,
            inference_ms=1.0,
        )

    def is_ready(self) -> bool:
        return self._loaded


class FailingDetector(BaseDetector):
    """Always fails on load — simulates transformer unavailable."""

    def load(self) -> None:
        raise RuntimeError("Model not found")

    def predict(self, text: str) -> DetectionResult:
        raise RuntimeError("Not loaded")

    def is_ready(self) -> bool:
        return False


# ── DetectionResult ───────────────────────────────────────────


class TestDetectionResult:
    def test_creation(self):
        r = DetectionResult(prob_ai=0.85, method="tfidf", inference_ms=12.3)
        assert r.prob_ai == 0.85
        assert r.method == "tfidf"
        assert r.metadata == {}

    def test_metadata(self):
        r = DetectionResult(
            prob_ai=0.5,
            method="test",
            inference_ms=1.0,
            metadata={"key": "value"},
        )
        assert r.metadata["key"] == "value"


# ── BaseDetector ──────────────────────────────────────────────


class TestBaseDetector:
    def test_get_info(self):
        d = FakeDetector()
        info = d.get_info()
        assert info["type"] == "FakeDetector"
        assert info["ready"] is False

    def test_get_info_after_load(self):
        d = FakeDetector()
        d.load()
        assert d.get_info()["ready"] is True


# ── CascadeDetector ──────────────────────────────────────────


class TestCascadeConfident:
    """When TF-IDF is confident (outside [lo, hi]), use fast path."""

    def test_high_confidence_ai(self):
        fast = FakeDetector(prob_ai=0.95, method="tfidf")
        slow = FakeDetector(prob_ai=0.99, method="transformer")
        fast.load()
        slow.load()
        cascade = CascadeDetector(fast, slow, cascade_lo=0.30, cascade_hi=0.70)

        result = cascade.predict("текст")
        assert result.method == "tfidf"
        assert result.metadata["cascade_path"] == "fast"

    def test_high_confidence_human(self):
        fast = FakeDetector(prob_ai=0.05, method="tfidf")
        slow = FakeDetector(prob_ai=0.10, method="transformer")
        fast.load()
        slow.load()
        cascade = CascadeDetector(fast, slow, cascade_lo=0.30, cascade_hi=0.70)

        result = cascade.predict("текст")
        assert result.method == "tfidf"
        assert result.metadata["cascade_path"] == "fast"

    def test_boundary_high_uses_fast(self):
        fast = FakeDetector(prob_ai=0.70, method="tfidf")
        slow = FakeDetector(prob_ai=0.80, method="transformer")
        fast.load()
        slow.load()
        cascade = CascadeDetector(fast, slow, cascade_lo=0.30, cascade_hi=0.70)

        result = cascade.predict("текст")
        assert result.metadata["cascade_path"] == "fast"

    def test_boundary_low_uses_fast(self):
        fast = FakeDetector(prob_ai=0.30, method="tfidf")
        slow = FakeDetector(prob_ai=0.25, method="transformer")
        fast.load()
        slow.load()
        cascade = CascadeDetector(fast, slow, cascade_lo=0.30, cascade_hi=0.70)

        result = cascade.predict("текст")
        assert result.metadata["cascade_path"] == "fast"


class TestCascadeUncertain:
    """When TF-IDF is uncertain (inside (lo, hi)), use slow path."""

    def test_uncertain_calls_transformer(self):
        fast = FakeDetector(prob_ai=0.50, method="tfidf")
        slow = FakeDetector(prob_ai=0.88, method="transformer")
        fast.load()
        slow.load()
        cascade = CascadeDetector(fast, slow, cascade_lo=0.30, cascade_hi=0.70)

        result = cascade.predict("текст")
        assert result.method == "transformer"
        assert result.metadata["cascade_path"] == "slow"
        assert result.metadata["fast_prob"] == 0.50

    def test_uncertain_sums_inference_time(self):
        fast = FakeDetector(prob_ai=0.50, method="tfidf")
        slow = FakeDetector(prob_ai=0.88, method="transformer")
        fast.load()
        slow.load()
        cascade = CascadeDetector(fast, slow)

        result = cascade.predict("текст")
        assert result.inference_ms == 2.0  # 1.0 + 1.0

    def test_just_above_lo(self):
        fast = FakeDetector(prob_ai=0.31, method="tfidf")
        slow = FakeDetector(prob_ai=0.75, method="transformer")
        fast.load()
        slow.load()
        cascade = CascadeDetector(fast, slow, cascade_lo=0.30, cascade_hi=0.70)

        result = cascade.predict("текст")
        assert result.metadata["cascade_path"] == "slow"


class TestCascadeFallback:
    """When Transformer is unavailable, fallback to TF-IDF."""

    def test_slow_none(self):
        fast = FakeDetector(prob_ai=0.50, method="tfidf")
        fast.load()
        cascade = CascadeDetector(fast, slow=None)

        result = cascade.predict("текст")
        assert result.method == "tfidf"
        assert result.metadata["cascade_path"] == "fallback"

    def test_slow_not_loaded(self):
        fast = FakeDetector(prob_ai=0.50, method="tfidf")
        slow = FakeDetector(prob_ai=0.88, method="transformer")
        fast.load()
        # slow NOT loaded → is_ready() == False
        cascade = CascadeDetector(fast, slow)

        result = cascade.predict("текст")
        assert result.metadata["cascade_path"] == "fallback"

    def test_slow_fails_on_load(self):
        fast = FakeDetector(prob_ai=0.50, method="tfidf")
        fast.load()
        cascade = CascadeDetector(fast, FailingDetector())
        cascade.load()  # should not raise

        result = cascade.predict("текст")
        assert result.metadata["cascade_path"] == "fallback"

    def test_fallback_warning_present(self):
        fast = FakeDetector(prob_ai=0.50, method="tfidf")
        fast.load()
        cascade = CascadeDetector(fast, slow=None)

        result = cascade.predict("текст")
        assert result.metadata.get("warning") == WarningCode.transformer_unavailable.value


class TestCascadeState:
    def test_is_ready_depends_on_fast(self):
        fast = FakeDetector()
        cascade = CascadeDetector(fast, slow=None)
        assert cascade.is_ready() is False

        fast.load()
        assert cascade.is_ready() is True

    def test_get_info(self):
        fast = FakeDetector()
        fast.load()
        cascade = CascadeDetector(fast, slow=None, cascade_lo=0.25, cascade_hi=0.75)
        info = cascade.get_info()

        assert info["type"] == "CascadeDetector"
        assert info["fast_ready"] is True
        assert info["slow_ready"] is False
        assert info["cascade_lo"] == 0.25
