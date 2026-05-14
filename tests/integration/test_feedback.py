"""Тесты эндпоинтов POST /api/v1/feedback и GET /api/v1/stats.

Используется реальный Postgres через testcontainers / TEST_DATABASE_URL —
консистентно с test_auth_api.py, test_batch_api.py, test_me_*. /stats требует
admin-роли — подменяем get_current_user через FastAPI dependency_overrides.

Замечание: предыдущая версия тестов мокировала UnitOfWork, что пропускало
ошибки на DB-уровне (например, CHECK constraint на feedback.source). Текущая
версия гарантирует, что эндпоинты совместимы со схемой БД.
"""

import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy import text as sql_text
from uuid_utils import uuid7

from app.auth.dependencies import get_current_user, get_current_user_optional
from app.database import engine as engine_mod
from app.database.models import Analysis, Feedback, User, UserRole, UserStatus
from app.database.uow import UnitOfWork
from app.routers.feedback import FeedbackRequest, FeedbackResponse
from tests.conftest import make_test_app

pytestmark = pytest.mark.asyncio(loop_scope="session")


# ── App + DB fixtures ─────────────────────────────────────────


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def app_with_engine(postgres_url, _engine):
    engine_mod.init_engine(postgres_url, pool_size=2, max_overflow=2)
    yield make_test_app(routers=("feedback",), db_enabled=True)
    await engine_mod.dispose_engine()


@pytest_asyncio.fixture(autouse=True, loop_scope="session")
async def _wipe_tables(app_with_engine):
    """Чистим feedback и analyses перед/после каждого теста — изолируем строки."""
    async with engine_mod.async_session() as s:
        for table in ("feedback", "analyses"):
            await s.execute(sql_text(f"DELETE FROM {table}"))
        await s.commit()
    yield
    async with engine_mod.async_session() as s:
        for table in ("feedback", "analyses"):
            await s.execute(sql_text(f"DELETE FROM {table}"))
        await s.commit()


@pytest_asyncio.fixture(loop_scope="session")
async def client(app_with_engine):
    transport = ASGITransport(app=app_with_engine)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── Helpers ───────────────────────────────────────────────────


def _override_user(app, *, admin: bool):
    """Подменить get_current_user возвращать синтетического пользователя.

    admin=True → роль ADMIN. admin=False → роль USER (для проверки 403 на /stats).
    Возвращает cleanup-функцию.
    """
    role = UserRole.ADMIN if admin else UserRole.USER
    fake_user = User(
        id=uuid.uuid4(),
        email=f"{role}@example.com",
        password_hash="$argon2id$dummy",
        status=UserStatus.ACTIVE,
        role=role,
    )
    app.dependency_overrides[get_current_user] = lambda: fake_user

    def _cleanup() -> None:
        app.dependency_overrides.pop(get_current_user, None)

    return _cleanup


# ── POST /api/v1/feedback ────────────────────────────────────


class TestSubmitFeedback:
    async def test_success_writes_row(self, client):
        resp = await client.post(
            "/api/v1/feedback",
            json={"text_hash": "abc123", "user_verdict": "human"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "saved"
        assert len(body["id"]) > 0  # uuid

        async with engine_mod.async_session() as s:
            rows = (await s.execute(select(Feedback))).scalars().all()
        assert len(rows) == 1
        assert rows[0].user_verdict == "human"
        assert rows[0].text_hash == "abc123"
        assert rows[0].source == "api"  # дефолт из FeedbackRequest

    async def test_ai_verdict_persisted(self, client):
        resp = await client.post(
            "/api/v1/feedback",
            json={"user_verdict": "ai", "comment": "Clearly generated"},
        )
        assert resp.status_code == 200

        async with engine_mod.async_session() as s:
            row = (await s.execute(select(Feedback))).scalar_one()
        assert row.user_verdict == "ai"
        assert row.comment == "Clearly generated"

    async def test_invalid_verdict_returns_422(self, client):
        resp = await client.post(
            "/api/v1/feedback", json={"user_verdict": "maybe"}
        )
        assert resp.status_code == 422

    async def test_missing_verdict_returns_422(self, client):
        resp = await client.post("/api/v1/feedback", json={})
        assert resp.status_code == 422

    async def test_db_disabled_returns_503(self, app_with_engine, client):
        app_with_engine.state.db_enabled = False
        try:
            resp = await client.post(
                "/api/v1/feedback", json={"user_verdict": "human"}
            )
            assert resp.status_code == 503
            assert "Database" in resp.json()["detail"]
        finally:
            app_with_engine.state.db_enabled = True

    async def test_with_analysis_id_persists_link(self, client):
        # Сначала создаём Analysis (через UoW — обходит UUID-sentinel issue
        # session.add() с uuid_utils.uuid7), потом feedback со ссылкой.
        analysis_id = uuid7()
        async with UnitOfWork() as uow:
            await uow.analyses.save(
                Analysis(
                    id=analysis_id,
                    request_id=uuid7(),
                    text_hash="a" * 64,
                    text_length=500,
                    status="SUCCESS",
                    verdict="ai",
                    confidence=0.85,
                    risk_level="HIGH",
                    detector_used="tfidf",
                    inference_ms=10.0,
                    cached=False,
                    model_version="v1",
                    service_version="0.1.0",
                    requested_at=datetime.now(timezone.utc),
                )
            )
            await uow.commit()

        resp = await client.post(
            "/api/v1/feedback",
            json={
                "analysis_id": str(analysis_id),
                "user_verdict": "ai",
                "source": "gradio",
                "comment": "Test feedback",
            },
        )
        assert resp.status_code == 200

        async with engine_mod.async_session() as s:
            row = (await s.execute(select(Feedback))).scalar_one()
        assert str(row.analysis_id) == str(analysis_id)
        assert row.source == "gradio"

    # NB: `source` ограничен CHECK-constraint'ом БД ('api','gradio','manual'),
    # но Pydantic-схема его не валидирует — невалидное значение приводит к
    # IntegrityError в роутере без маппинга на 400. Это код-баг роутера,
    # занесён в docs/ROADMAP.md → C-секция. Соответствующий тест добавим
    # после фикса (IntegrityError → 400) — в текущем состоянии тест ловил бы
    # неудобный 5xx и был бы хрупким.


class TestFeedbackOwnership:
    """Feedback на чужой анализ возвращает 403."""

    @staticmethod
    async def _seed_user_and_analysis(
        owner_id: uuid.UUID | None, *, email_suffix: str
    ) -> uuid.UUID:
        """Создать User (если owner_id указан) и Analysis с user_id=owner_id."""
        async with UnitOfWork() as uow:
            if owner_id is not None:
                await uow.users.create(
                    User(
                        id=owner_id,
                        email=f"owner-{email_suffix}@example.com",
                        password_hash="$argon2id$dummy",
                        status=UserStatus.ACTIVE,
                        role=UserRole.USER,
                    )
                )
            analysis_id = uuid7()
            await uow.analyses.save(
                Analysis(
                    id=analysis_id,
                    request_id=uuid7(),
                    text_hash="o" * 64,
                    text_length=500,
                    status="SUCCESS",
                    verdict="ai",
                    confidence=0.85,
                    risk_level="HIGH",
                    detector_used="tfidf",
                    inference_ms=10.0,
                    cached=False,
                    model_version="v1",
                    service_version="0.1.0",
                    user_id=owner_id,
                    requested_at=datetime.now(timezone.utc),
                )
            )
            await uow.commit()
        return analysis_id

    async def test_owner_can_submit_feedback(self, app_with_engine, client):
        owner_id = uuid.uuid4()
        analysis_id = await self._seed_user_and_analysis(owner_id, email_suffix="own")

        owner = User(
            id=owner_id, email="owner-own@example.com",
            password_hash="$argon2id$dummy",
            status=UserStatus.ACTIVE, role=UserRole.USER,
        )
        app_with_engine.dependency_overrides[get_current_user_optional] = lambda: owner
        try:
            resp = await client.post(
                "/api/v1/feedback",
                json={"analysis_id": str(analysis_id), "user_verdict": "human"},
            )
            assert resp.status_code == 200
        finally:
            app_with_engine.dependency_overrides.pop(get_current_user_optional, None)

    async def test_other_user_gets_403(self, app_with_engine, client):
        owner_id = uuid.uuid4()
        analysis_id = await self._seed_user_and_analysis(owner_id, email_suffix="other")

        intruder_id = uuid.uuid4()
        async with UnitOfWork() as uow:
            await uow.users.create(
                User(
                    id=intruder_id, email="intruder@example.com",
                    password_hash="$argon2id$dummy",
                    status=UserStatus.ACTIVE, role=UserRole.USER,
                )
            )
            await uow.commit()
        intruder = User(
            id=intruder_id, email="intruder@example.com",
            password_hash="$argon2id$dummy",
            status=UserStatus.ACTIVE, role=UserRole.USER,
        )
        app_with_engine.dependency_overrides[get_current_user_optional] = lambda: intruder
        try:
            resp = await client.post(
                "/api/v1/feedback",
                json={"analysis_id": str(analysis_id), "user_verdict": "human"},
            )
            assert resp.status_code == 403
            assert "чужой" in resp.json()["detail"].lower()
        finally:
            app_with_engine.dependency_overrides.pop(get_current_user_optional, None)

    async def test_anonymous_user_on_owned_analysis_gets_403(self, client):
        analysis_id = await self._seed_user_and_analysis(uuid.uuid4(), email_suffix="anon")
        # Без override: get_current_user_optional возвращает None (anonymous).
        resp = await client.post(
            "/api/v1/feedback",
            json={"analysis_id": str(analysis_id), "user_verdict": "human"},
        )
        assert resp.status_code == 403

    async def test_anonymous_user_on_anonymous_analysis_allowed(self, client):
        """analysis.user_id is None → feedback анонимный без auth разрешён."""
        analysis_id = await self._seed_user_and_analysis(None, email_suffix="none")
        resp = await client.post(
            "/api/v1/feedback",
            json={"analysis_id": str(analysis_id), "user_verdict": "ai"},
        )
        assert resp.status_code == 200

    async def test_invalid_uuid_in_analysis_id_does_not_break(self, client):
        """Невалидный UUID не должен ронять 500 — просто сохраняем без привязки."""
        resp = await client.post(
            "/api/v1/feedback",
            json={"analysis_id": "not-a-uuid", "user_verdict": "ai"},
        )
        assert resp.status_code == 200


# ── GET /api/v1/stats ────────────────────────────────────────


class TestGetStats:
    async def test_admin_gets_aggregates(self, app_with_engine, client):
        # Seed: 2 анализа (ai high + human low) — через UoW для совместимости
        # с asyncpg + uuid_utils.uuid7 (session.add даёт UUID-sentinel issue).
        async with UnitOfWork() as uow:
            for verdict, conf, risk in [("ai", 0.9, "HIGH"), ("human", 0.2, "LOW")]:
                await uow.analyses.save(
                    Analysis(
                        id=uuid7(),
                        request_id=uuid7(),
                        text_hash="x" * 64,
                        text_length=500,
                        status="SUCCESS",
                        verdict=verdict,
                        confidence=conf,
                        risk_level=risk,
                        detector_used="tfidf",
                        inference_ms=10.0,
                        cached=False,
                        model_version="v1",
                        service_version="0.1.0",
                        requested_at=datetime.now(timezone.utc),
                    )
                )
            await uow.commit()

        cleanup = _override_user(app_with_engine, admin=True)
        try:
            resp = await client.get("/api/v1/stats")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total"] == 2
        finally:
            cleanup()

    async def test_unauthenticated_returns_4xx(self, client):
        # Без токена и без override — FastAPI HTTPBearer возвращает 401/403.
        resp = await client.get("/api/v1/stats")
        assert resp.status_code in (401, 403)

    async def test_non_admin_returns_403(self, app_with_engine, client):
        cleanup = _override_user(app_with_engine, admin=False)
        try:
            resp = await client.get("/api/v1/stats")
            assert resp.status_code == 403
            assert "admin" in resp.json()["detail"].lower()
        finally:
            cleanup()

    async def test_db_disabled_returns_503_for_admin(self, app_with_engine, client):
        cleanup = _override_user(app_with_engine, admin=True)
        app_with_engine.state.db_enabled = False
        try:
            resp = await client.get("/api/v1/stats")
            assert resp.status_code == 503
        finally:
            app_with_engine.state.db_enabled = True
            cleanup()


# ── Schema-level tests (pure unit, без БД) ───────────────────


class TestFeedbackSchemas:
    def test_human_verdict_valid(self):
        req = FeedbackRequest(user_verdict="human")
        assert req.user_verdict == "human"
        assert req.source == "api"

    def test_ai_verdict_with_comment(self):
        req = FeedbackRequest(user_verdict="ai", comment="test")
        assert req.user_verdict == "ai"
        assert req.comment == "test"

    def test_unknown_verdict_raises(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(user_verdict="unknown")

    def test_response_default_status(self):
        resp = FeedbackResponse(id="abc-123")
        assert resp.status == "saved"
        assert resp.id == "abc-123"
