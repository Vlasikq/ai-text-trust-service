"""Tests for database models — schema creation, constraints, JSONB."""

from datetime import datetime, timezone

import pytest
from uuid_utils import uuid7

from app.database.models import Analysis, Feedback, ModelDeployment

pytestmark = pytest.mark.asyncio(loop_scope="session")


# ── Model Deployment ─────────────────────────────────────────


class TestModelDeployment:
    async def test_create(self, db_session):
        dep = ModelDeployment(
            model_version="ru-detector-0.1.0",
            service_version="0.1.0",
            detector_type="cascade",
        )
        db_session.add(dep)
        await db_session.flush()

        assert dep.id is not None
        assert dep.deployed_at is not None

    async def test_config_snapshot_jsonb(self, db_session):
        dep = ModelDeployment(
            model_version="v1",
            service_version="0.1.0",
            config_snapshot={"cascade_lo": 0.30, "cascade_hi": 0.70},
        )
        db_session.add(dep)
        await db_session.flush()

        result = await db_session.get(ModelDeployment, dep.id)
        assert result.config_snapshot["cascade_lo"] == 0.30


# ── Analysis ──────────────────────────────────────────────────


class TestAnalysis:
    async def test_create_success(self, db_session):
        a = Analysis(
            request_id=uuid7(),
            text_hash="abc123" * 10 + "abcd",
            text_length=500,
            status="SUCCESS",
            verdict="ai",
            confidence=0.85,
            risk_level="HIGH",
            detector_used="tfidf",
            cascade_path="fast",
            inference_ms=12.5,
            model_version="v1",
            service_version="0.1.0",
            warnings=["TEXT_TRUNCATED"],
            requested_at=datetime.now(timezone.utc),
        )
        db_session.add(a)
        await db_session.flush()

        assert a.id is not None
        assert a.persisted_at is not None

    async def test_create_no_decision(self, db_session):
        a = Analysis(
            request_id=uuid7(),
            text_hash="def456" * 10 + "defg",
            text_length=50,
            status="NO_DECISION",
            verdict=None,
            confidence=None,
            risk_level=None,
            detector_used="tfidf",
            inference_ms=1.0,
            model_version="v1",
            service_version="0.1.0",
            warnings=["TEXT_TOO_SHORT"],
            requested_at=datetime.now(timezone.utc),
        )
        db_session.add(a)
        await db_session.flush()

        assert a.verdict is None
        assert a.risk_level is None

    async def test_warnings_jsonb(self, db_session):
        a = Analysis(
            request_id=uuid7(),
            text_hash="w" * 64,
            text_length=500,
            status="SUCCESS",
            verdict="ai",
            confidence=0.5,
            risk_level="MEDIUM",
            detector_used="cascade",
            cascade_path="fallback",
            inference_ms=10.0,
            model_version="v1",
            service_version="0.1.0",
            warnings=["TRANSFORMER_UNAVAILABLE", "LOW_CONFIDENCE"],
            requested_at=datetime.now(timezone.utc),
        )
        db_session.add(a)
        await db_session.flush()

        result = await db_session.get(Analysis, a.id)
        assert "TRANSFORMER_UNAVAILABLE" in result.warnings
        assert len(result.warnings) == 2

    async def test_explanation_jsonb(self, db_session):
        explanation = {
            "top_markers": [
                {
                    "feature": "avg_word_len",
                    "value": 6.2,
                    "human_baseline": 5.1,
                    "deviation_percent": 21.6,
                    "direction": "above",
                }
            ],
            "summary": "Ключевые отклонения: средняя длина слова.",
        }
        a = Analysis(
            request_id=uuid7(),
            text_hash="e" * 64,
            text_length=500,
            status="SUCCESS",
            verdict="ai",
            confidence=0.9,
            risk_level="HIGH",
            detector_used="tfidf",
            inference_ms=5.0,
            model_version="v1",
            service_version="0.1.0",
            explanation=explanation,
            requested_at=datetime.now(timezone.utc),
        )
        db_session.add(a)
        await db_session.flush()

        result = await db_session.get(Analysis, a.id)
        assert result.explanation["top_markers"][0]["feature"] == "avg_word_len"

    async def test_request_id_unique(self, db_session):
        rid = uuid7()
        a1 = Analysis(
            request_id=rid,
            text_hash="u1" * 32,
            text_length=500,
            status="SUCCESS",
            verdict="ai",
            confidence=0.9,
            risk_level="HIGH",
            detector_used="tfidf",
            inference_ms=5.0,
            model_version="v1",
            service_version="0.1.0",
            requested_at=datetime.now(timezone.utc),
        )
        db_session.add(a1)
        await db_session.flush()

        a2 = Analysis(
            request_id=rid,  # same request_id
            text_hash="u2" * 32,
            text_length=500,
            status="SUCCESS",
            verdict="human",
            confidence=0.1,
            risk_level="LOW",
            detector_used="tfidf",
            inference_ms=5.0,
            model_version="v1",
            service_version="0.1.0",
            requested_at=datetime.now(timezone.utc),
        )
        db_session.add(a2)
        with pytest.raises(Exception):  # IntegrityError
            await db_session.flush()


# ── Feedback ──────────────────────────────────────────────────


class TestFeedback:
    async def test_create_with_analysis(self, db_session):
        a = Analysis(
            request_id=uuid7(),
            text_hash="f" * 64,
            text_length=500,
            status="SUCCESS",
            verdict="ai",
            confidence=0.9,
            risk_level="HIGH",
            detector_used="tfidf",
            inference_ms=5.0,
            model_version="v1",
            service_version="0.1.0",
            requested_at=datetime.now(timezone.utc),
        )
        db_session.add(a)
        await db_session.flush()

        fb = Feedback(
            analysis_id=a.id,
            text_hash="f" * 64,
            user_verdict="human",
            source="gradio",
            comment="Это точно человек написал.",
        )
        db_session.add(fb)
        await db_session.flush()

        assert fb.id is not None
        assert fb.created_at is not None

    async def test_create_without_analysis(self, db_session):
        fb = Feedback(
            text_hash="g" * 64,
            user_verdict="ai",
            source="manual",
        )
        db_session.add(fb)
        await db_session.flush()

        assert fb.analysis_id is None
