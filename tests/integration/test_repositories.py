"""Тесты репозиториев: CRUD, идемпотентность, агрегаты для статистики.

Нужен PostgreSQL (testcontainers или TEST_DATABASE_URL). Если БД недоступна —
тесты аккуратно скипаются.
"""

from datetime import datetime, timezone

import pytest
from uuid_utils import uuid7

from app.database.models import Analysis, Feedback, ModelDeployment
from app.database.repositories import (
    AnalysisRepository,
    DeploymentRepository,
    FeedbackRepository,
)

pytestmark = pytest.mark.asyncio(loop_scope="session")


def _make_analysis(**overrides) -> Analysis:
    defaults = {
        "id": uuid7(),
        "request_id": uuid7(),
        "text_hash": "a" * 64,
        "text_length": 500,
        "status": "SUCCESS",
        "verdict": "ai",
        "confidence": 0.85,
        "risk_level": "HIGH",
        "detector_used": "tfidf",
        "cascade_path": "fast",
        "inference_ms": 12.0,
        "cached": False,
        "model_version": "v1",
        "service_version": "0.1.0",
        "warnings": [],
        "requested_at": datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    return Analysis(**defaults)


# ── AnalysisRepository ────────────────────────────────────────


class TestAnalysisRepo:
    async def test_save_and_get(self, db_session):
        repo = AnalysisRepository(db_session)
        a = _make_analysis()
        inserted = await repo.save(a)
        await db_session.flush()
        assert inserted is True

        found = await repo.get_by_request_id(a.request_id)
        assert found is not None
        assert found.confidence == 0.85

    async def test_idempotent_save(self, db_session):
        repo = AnalysisRepository(db_session)
        rid = uuid7()
        a1 = _make_analysis(request_id=rid, id=uuid7())
        a2 = _make_analysis(request_id=rid, id=uuid7(), confidence=0.99)

        r1 = await repo.save(a1)
        await db_session.flush()
        r2 = await repo.save(a2)
        await db_session.flush()

        assert r1 is True
        assert r2 is False  # ON CONFLICT DO NOTHING

        found = await repo.get_by_request_id(rid)
        assert found.confidence == 0.85  # first write wins

    async def test_get_by_text_hash(self, db_session):
        repo = AnalysisRepository(db_session)
        th = "h" * 64
        a1 = _make_analysis(text_hash=th)
        a2 = _make_analysis(text_hash=th)

        await repo.save(a1)
        await repo.save(a2)
        await db_session.flush()

        results = await repo.get_by_text_hash(th)
        assert len(results) == 2

    async def test_get_by_request_id_not_found(self, db_session):
        repo = AnalysisRepository(db_session)
        result = await repo.get_by_request_id(uuid7())
        assert result is None

    async def test_get_stats(self, db_session):
        repo = AnalysisRepository(db_session)
        await repo.save(_make_analysis(risk_level="HIGH", detector_used="tfidf"))
        await repo.save(_make_analysis(risk_level="HIGH", detector_used="tfidf"))
        await repo.save(
            _make_analysis(risk_level="LOW", detector_used="cascade", cascade_path="slow")
        )
        await db_session.flush()

        stats = await repo.get_stats()
        assert stats["total"] == 3
        assert stats["by_risk_level"]["HIGH"] == 2
        assert stats["by_detector"]["tfidf"] == 2

    async def test_warnings_stored(self, db_session):
        repo = AnalysisRepository(db_session)
        a = _make_analysis(warnings=["TEXT_TRUNCATED", "LOW_CONFIDENCE"])
        await repo.save(a)
        await db_session.flush()

        found = await repo.get_by_request_id(a.request_id)
        assert "TEXT_TRUNCATED" in found.warnings


# ── FeedbackRepository ────────────────────────────────────────


class TestFeedbackRepo:
    async def test_save_with_analysis(self, db_session):
        # Need an analysis first
        a_repo = AnalysisRepository(db_session)
        a = _make_analysis()
        await a_repo.save(a)
        await db_session.flush()

        fb_repo = FeedbackRepository(db_session)
        fb = Feedback(
            analysis_id=a.id,
            text_hash=a.text_hash,
            user_verdict="human",
            source="gradio",
        )
        await fb_repo.save(fb)
        await db_session.flush()

        results = await fb_repo.get_by_analysis_id(a.id)
        assert len(results) == 1
        assert results[0].user_verdict == "human"

    async def test_save_without_analysis(self, db_session):
        fb_repo = FeedbackRepository(db_session)
        fb = Feedback(
            text_hash="x" * 64,
            user_verdict="ai",
            source="manual",
        )
        await fb_repo.save(fb)
        await db_session.flush()

        results = await fb_repo.get_by_text_hash("x" * 64)
        assert len(results) >= 1


# ── DeploymentRepository ──────────────────────────────────────


class TestDeploymentRepo:
    async def test_save_and_get_latest(self, db_session):
        repo = DeploymentRepository(db_session)
        dep = ModelDeployment(
            model_version="v1.0",
            service_version="0.1.0",
            detector_type="cascade",
        )
        await repo.save(dep)
        await db_session.flush()

        latest = await repo.get_latest()
        assert latest is not None
        assert latest.model_version == "v1.0"
