"""Юнит-тесты для services.detection.run_detection — единая точка входа для
sync-роутера и async-worker'а. Контракт фиксируется здесь, а не косвенно
через test_api.py: timeout, cache, calibration, no-decision, explain, debug.
"""

from __future__ import annotations

import logging

import pytest

from app.cache import DetectionCache
from app.config import Settings
from app.detectors.base import BaseDetector
from app.preprocessing.pipeline import TextPreprocessor
from app.schemas import (
    RiskLevel,
    Status,
    Verdict,
    WarningCode,
)
from app.services.detection import DetectionContext, run_detection
from tests.conftest import FakeDetector, FakeExplainer

pytestmark = pytest.mark.asyncio(loop_scope="function")


class StubCalibrator:
    """Минимальный стаб ProbabilityCalibrator: возвращает фиксированное значение."""

    def __init__(self, return_value: float = 0.95):
        self._return_value = return_value
        self.calls: list[tuple[float, str]] = []

    def calibrate(self, prob_ai: float, method: str) -> float:
        self.calls.append((prob_ai, method))
        return self._return_value


def _make_ctx(
    *,
    detector: BaseDetector | None = None,
    calibrator: object | None = None,
    explainer: object | None = None,
    cache: DetectionCache | None = None,
    min_chars: int = 10,
    max_chars: int = 8000,
    risk_lo: float = 0.30,
    risk_hi: float = 0.70,
    verdict_threshold: float = 0.50,
) -> DetectionContext:
    settings = Settings(
        min_chars=min_chars,
        max_chars=max_chars,
        risk_thresh_low=risk_lo,
        risk_thresh_high=risk_hi,
        verdict_threshold=verdict_threshold,
    )
    return DetectionContext(
        settings=settings,
        preprocessor=TextPreprocessor(settings),
        detector=detector or FakeDetector(),
        calibrator=calibrator,  # type: ignore[arg-type]
        explainer=explainer,  # type: ignore[arg-type]
        cache=cache,
    )


_REQUEST_ID = "00000000-0000-4000-8000-000000000001"


# ── Happy path ────────────────────────────────────────────────


class TestRunDetectionHappyPath:
    async def test_returns_success_with_verdict_and_risk(self):
        ctx = _make_ctx(detector=FakeDetector(prob_ai=0.85))
        outcome = await run_detection(ctx, "Длинный текст для анализа.", request_id=_REQUEST_ID)

        assert outcome.response.status == Status.success
        assert outcome.response.verdict == Verdict.ai
        assert outcome.response.risk_level == RiskLevel.high
        assert outcome.response.confidence == 0.85
        assert outcome.detection is not None
        assert outcome.detection.method == "tfidf"
        assert outcome.cached_hit is False

    async def test_low_prob_returns_human_low_risk(self):
        ctx = _make_ctx(detector=FakeDetector(prob_ai=0.10))
        outcome = await run_detection(ctx, "Длинный текст для анализа.", request_id=_REQUEST_ID)

        assert outcome.response.verdict == Verdict.human
        assert outcome.response.risk_level == RiskLevel.low

    async def test_medium_risk_adds_low_confidence_warning(self):
        ctx = _make_ctx(detector=FakeDetector(prob_ai=0.50))
        outcome = await run_detection(ctx, "Длинный текст для анализа.", request_id=_REQUEST_ID)

        assert outcome.response.risk_level == RiskLevel.medium
        assert WarningCode.low_confidence in outcome.response.warnings


# ── No-decision / truncation ─────────────────────────────────


class TestRunDetectionPreprocessing:
    async def test_too_short_text_returns_no_decision(self):
        ctx = _make_ctx(min_chars=300)
        outcome = await run_detection(ctx, "Слишком короткий.", request_id=_REQUEST_ID)

        assert outcome.response.status == Status.no_decision
        assert outcome.response.verdict is None
        assert WarningCode.text_too_short in outcome.response.warnings
        # При no-decision сырой результат детектора не нужен — детектор не вызывался.
        assert outcome.detection is None

    async def test_too_long_text_truncated_with_warning(self):
        ctx = _make_ctx(min_chars=10, max_chars=50)
        long_text = "Первое предложение. Второе предложение. Третье. Четвёртое предложение здесь."
        outcome = await run_detection(ctx, long_text, request_id=_REQUEST_ID)

        assert outcome.response.was_truncated is True
        assert WarningCode.text_truncated in outcome.response.warnings


# ── Cache ─────────────────────────────────────────────────────


class TestRunDetectionCache:
    async def test_first_call_misses_second_hits(self):
        cache = DetectionCache(maxsize=10)
        # Считаем вызовы predict через счётчик в стабе.
        call_count = {"n": 0}

        class CountingDetector(FakeDetector):
            def predict(self, text: str) -> DetectionResult:
                call_count["n"] += 1
                return super().predict(text)

        ctx = _make_ctx(detector=CountingDetector(prob_ai=0.85), cache=cache)

        text = "Один и тот же текст для проверки кэша."
        first = await run_detection(ctx, text, request_id=_REQUEST_ID)
        second = await run_detection(ctx, text, request_id=_REQUEST_ID)

        assert call_count["n"] == 1, "второй вызов должен прийти из кэша"
        assert first.cached_hit is False
        assert second.cached_hit is True
        assert first.response.confidence == second.response.confidence


# ── Calibration ───────────────────────────────────────────────


class TestRunDetectionCalibration:
    async def test_calibrator_applied_when_present(self):
        cal = StubCalibrator(return_value=0.95)
        ctx = _make_ctx(detector=FakeDetector(prob_ai=0.60), calibrator=cal)

        outcome = await run_detection(ctx, "Длинный текст для анализа.", request_id=_REQUEST_ID)

        assert outcome.response.confidence == 0.95
        assert cal.calls == [(0.60, "tfidf")]

    async def test_calibrator_failure_fallbacks_to_raw(self):
        class FailingCalibrator:
            def calibrate(self, prob_ai, method):
                raise RuntimeError("calibrator broken")

        ctx = _make_ctx(
            detector=FakeDetector(prob_ai=0.80),
            calibrator=FailingCalibrator(),
        )
        outcome = await run_detection(ctx, "Длинный текст для анализа.", request_id=_REQUEST_ID)

        assert outcome.response.confidence == 0.80
        assert WarningCode.calibration_unavailable in outcome.response.warnings

    async def test_calibrator_failure_logs_warning_with_traceback(self, caplog):
        """PR-55: при сбое калибровки должна остаться запись WARNING со stack trace
        и request_id, чтобы на проде можно было разобрать причину через journald/Sentry.
        """

        class FailingCalibrator:
            def calibrate(self, prob_ai, method):
                raise RuntimeError("calibrator broken")

        ctx = _make_ctx(
            detector=FakeDetector(prob_ai=0.80),
            calibrator=FailingCalibrator(),
        )

        with caplog.at_level(logging.WARNING, logger="app.services.detection"):
            await run_detection(ctx, "Длинный текст для анализа.", request_id=_REQUEST_ID)

        matching = [
            r for r in caplog.records
            if r.name == "app.services.detection" and r.levelname == "WARNING"
        ]
        assert matching, "ожидался WARNING-лог о падении калибровки"
        assert "Calibration failed" in matching[0].getMessage()
        assert _REQUEST_ID in matching[0].getMessage()
        # exc_info=True сохраняет тройку (type, value, traceback) в record.exc_info.
        assert matching[0].exc_info is not None


# ── Timeout ───────────────────────────────────────────────────


class TestRunDetectionTimeout:
    async def test_inference_timeout_returns_error(self):
        # Детектор «спит» 0.5с, лимит 0.05с — гарантированный timeout.
        ctx = _make_ctx(detector=FakeDetector(prob_ai=0.85, sleep_s=0.5))

        outcome = await run_detection(
            ctx, "Длинный текст для анализа.", request_id=_REQUEST_ID, timeout_s=0.05
        )

        assert outcome.response.status == Status.error
        assert WarningCode.inference_timeout in outcome.response.warnings
        assert outcome.detection is None

    async def test_no_timeout_when_none(self):
        # Worker передаёт timeout_s=None → ждём сколько надо.
        ctx = _make_ctx(detector=FakeDetector(prob_ai=0.85, sleep_s=0.05))

        outcome = await run_detection(
            ctx, "Длинный текст для анализа.", request_id=_REQUEST_ID, timeout_s=None
        )

        assert outcome.response.status == Status.success


# ── Explain / debug ───────────────────────────────────────────


class TestRunDetectionExplainAndDebug:
    async def test_explain_adds_explanation_and_warning(self):
        ctx = _make_ctx(detector=FakeDetector(prob_ai=0.85), explainer=FakeExplainer())

        outcome = await run_detection(
            ctx, "Длинный текст для анализа.", request_id=_REQUEST_ID, explain=True
        )

        assert outcome.response.explanation is not None
        assert outcome.response.explanation.summary
        assert WarningCode.style_explanation_heuristic in outcome.response.warnings

    async def test_explainer_failure_does_not_break_response(self):
        ctx = _make_ctx(
            detector=FakeDetector(prob_ai=0.85),
            explainer=FakeExplainer(raises=RuntimeError("broken explainer")),
        )

        outcome = await run_detection(
            ctx, "Длинный текст для анализа.", request_id=_REQUEST_ID, explain=True
        )

        assert outcome.response.status == Status.success
        assert outcome.response.explanation is None

    async def test_debug_adds_method_scores(self):
        ctx = _make_ctx(detector=FakeDetector(prob_ai=0.85, method="tfidf"))

        outcome = await run_detection(
            ctx, "Длинный текст для анализа.", request_id=_REQUEST_ID, debug=True
        )

        assert outcome.response.method_scores is not None
        assert len(outcome.response.method_scores) == 1
        assert outcome.response.method_scores[0].method == "tfidf"
        assert outcome.response.method_scores[0].raw_score == 0.85


# ── Cascade fallback warning ──────────────────────────────────


class TestRunDetectionCascadeMetadata:
    async def test_cascade_fallback_warning_propagates(self):
        # CascadeDetector выставляет metadata["warning"]="TRANSFORMER_UNAVAILABLE"
        # когда трансформер не загружен — services должен пробросить в response.
        ctx = _make_ctx(
            detector=FakeDetector(
                prob_ai=0.85,
                metadata={"warning": WarningCode.transformer_unavailable.value},
            ),
        )

        outcome = await run_detection(ctx, "Длинный текст для анализа.", request_id=_REQUEST_ID)

        assert WarningCode.transformer_unavailable in outcome.response.warnings
