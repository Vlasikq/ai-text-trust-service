"""Tests for app.schemas — request/response validation."""

import pytest
from pydantic import ValidationError

from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    Explanation,
    MethodScore,
    RiskLevel,
    Status,
    StyleMarker,
    Verdict,
    WarningCode,
)

# ── AnalyzeRequest ────────────────────────────────────────────


class TestAnalyzeRequest:
    def test_minimal_request(self):
        r = AnalyzeRequest(text="Привет мир")
        assert r.text == "Привет мир"
        assert r.explain is False
        assert r.debug is False

    def test_all_flags(self):
        r = AnalyzeRequest(text="Текст", explain=True, debug=True)
        assert r.explain is True
        assert r.debug is True

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            AnalyzeRequest(text="")

    def test_too_long_text_rejected(self):
        with pytest.raises(ValidationError):
            AnalyzeRequest(text="x" * 50_001)

    def test_max_length_text_accepted(self):
        r = AnalyzeRequest(text="x" * 50_000)
        assert len(r.text) == 50_000

    def test_missing_text_rejected(self):
        with pytest.raises(ValidationError):
            AnalyzeRequest()


# ── AnalyzeResponse ───────────────────────────────────────────


class TestAnalyzeResponse:
    def _make_response(self, **overrides):
        defaults = {
            "request_id": "test-123",
            "status": Status.success,
            "verdict": Verdict.ai,
            "confidence": 0.85,
            "risk_level": RiskLevel.high,
            "text_length": 500,
            "processing_time_ms": 42.0,
        }
        defaults.update(overrides)
        return AnalyzeResponse(**defaults)

    def test_success_response(self):
        r = self._make_response()
        assert r.status == Status.success
        assert r.verdict == Verdict.ai
        assert r.confidence == 0.85
        assert r.risk_level == RiskLevel.high

    def test_no_decision_response(self):
        r = self._make_response(
            status=Status.no_decision,
            verdict=None,
            confidence=None,
            risk_level=None,
            warnings=[WarningCode.text_too_short],
        )
        assert r.status == Status.no_decision
        assert r.verdict is None
        assert WarningCode.text_too_short in r.warnings

    def test_disclaimer_present(self):
        r = self._make_response()
        assert "вероятностный" in r.disclaimer

    def test_default_warnings_empty(self):
        r = self._make_response()
        assert r.warnings == []

    def test_was_truncated_default_false(self):
        r = self._make_response()
        assert r.was_truncated is False

    def test_explanation_optional(self):
        r = self._make_response(explanation=None)
        assert r.explanation is None

    def test_method_scores_optional(self):
        r = self._make_response(method_scores=None)
        assert r.method_scores is None

    def test_serialization_roundtrip(self):
        r = self._make_response()
        data = r.model_dump()
        r2 = AnalyzeResponse(**data)
        assert r2.request_id == r.request_id
        assert r2.confidence == r.confidence


# ── Sub-models ────────────────────────────────────────────────


class TestStyleMarker:
    def test_creation(self):
        m = StyleMarker(
            feature="avg_word_len",
            value=6.2,
            human_baseline=5.1,
            deviation_percent=21.6,
            direction="above",
        )
        assert m.feature == "avg_word_len"
        assert m.direction == "above"


class TestExplanation:
    def test_creation(self):
        marker = StyleMarker(
            feature="ttr",
            value=0.8,
            human_baseline=0.65,
            deviation_percent=23.1,
            direction="above",
        )
        e = Explanation(
            top_markers=[marker],
            summary="Текст имеет нетипично высокий TTR.",
        )
        assert len(e.top_markers) == 1
        assert "TTR" in e.summary


class TestMethodScore:
    def test_creation(self):
        ms = MethodScore(method="tfidf", raw_score=0.92, inference_ms=12.3)
        assert ms.method == "tfidf"
        assert ms.calibrated_score is None

    def test_with_calibrated(self):
        ms = MethodScore(
            method="transformer",
            raw_score=0.88,
            calibrated_score=0.91,
            inference_ms=450.0,
        )
        assert ms.calibrated_score == 0.91


# ── Enums ─────────────────────────────────────────────────────


class TestEnums:
    def test_risk_levels(self):
        assert RiskLevel.high.value == "HIGH"
        assert RiskLevel.medium.value == "MEDIUM"
        assert RiskLevel.low.value == "LOW"

    def test_verdict_values(self):
        assert Verdict.human.value == "human"
        assert Verdict.ai.value == "ai"

    def test_status_values(self):
        assert set(s.value for s in Status) == {"SUCCESS", "NO_DECISION", "ERROR"}

    def test_warning_codes_complete(self):
        expected = {
            "TEXT_TOO_SHORT",
            "TEXT_TRUNCATED",
            "TRANSFORMER_UNAVAILABLE",
            "LOW_CONFIDENCE",
            "CALIBRATION_UNAVAILABLE",
            "INFERENCE_TIMEOUT",
            "STYLE_EXPLANATION_HEURISTIC",
        }
        assert set(w.value for w in WarningCode) == expected
