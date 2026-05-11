"""Тесты GET /me/analyses/{id} — detailed scan view с owner-check.

Контракт:
  - Owner видит свою запись → 200 + полная карточка (explanation, warnings, cascade_path).
  - Чужая запись → 404 (НЕ 403 — иначе enumeration).
  - Несуществующая → 404.
  - Невалидный UUID → 404 (одинаково с «не найдено» — не утекаем формат id).
  - Без auth → 401 (HTTPBearer auto_error).
"""

from datetime import datetime, timezone

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text as sql_text
from uuid_utils import uuid7

from app.database import engine as engine_mod
from app.database.models import Analysis
from app.database.uow import UnitOfWork
from tests.conftest import make_test_app

pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def app_with_engine(postgres_url, _engine):
    engine_mod.init_engine(postgres_url, pool_size=2, max_overflow=2)
    yield make_test_app(routers=("auth", "me"))
    await engine_mod.dispose_engine()


@pytest_asyncio.fixture(autouse=True, loop_scope="session")
async def _wipe_tables(app_with_engine):
    async with engine_mod.async_session() as s:
        for table in ("refresh_tokens", "analyses", "users"):
            await s.execute(sql_text(f"DELETE FROM {table}"))
        await s.commit()
    yield
    async with engine_mod.async_session() as s:
        for table in ("refresh_tokens", "analyses", "users"):
            await s.execute(sql_text(f"DELETE FROM {table}"))
        await s.commit()


@pytest_asyncio.fixture(loop_scope="session")
async def client(app_with_engine):
    transport = ASGITransport(app=app_with_engine)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def _register(client: AsyncClient, email: str = "owner@example.com") -> dict:
    r = await client.post(
        "/auth/register", json={"email": email, "password": "supersecret123"}
    )
    return r.json()


async def _make_analysis(
    user_id: str | None,
    *,
    explanation: dict | None = None,
    warnings: list[str] | None = None,
) -> str:
    """Сохранить тестовый Analysis. Возвращает str(id)."""
    aid = uuid7()
    async with UnitOfWork() as uow:
        await uow.analyses.save(
            Analysis(
                id=aid,
                request_id=uuid7(),
                text_hash="a" * 64,
                text_length=512,
                status="SUCCESS",
                verdict="ai",
                confidence=0.83,
                risk_level="HIGH",
                detector_used="cascade",
                cascade_path="slow",
                inference_ms=42.5,
                cached=False,
                model_version="v1",
                service_version="0.1.0",
                user_id=user_id,
                warnings=warnings or [],
                explanation=explanation,
                requested_at=datetime.now(timezone.utc),
            )
        )
        await uow.commit()
    return str(aid)


class TestDetailEndpoint:
    async def test_owner_gets_full_card(self, client):
        body = await _register(client, email="alice@example.com")
        access = body["tokens"]["access_token"]
        user_id = body["user"]["id"]

        explanation = {
            "summary": "Стиль шаблонный.",
            "top_markers": [
                {
                    "feature": "ttr",
                    "value": 0.42,
                    "human_baseline": 0.55,
                    "deviation_percent": -23.6,
                    "direction": "below",
                }
            ],
        }
        aid = await _make_analysis(
            user_id, explanation=explanation, warnings=["LOW_CONFIDENCE"]
        )

        resp = await client.get(
            f"/me/analyses/{aid}", headers={"Authorization": f"Bearer {access}"}
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["id"] == aid
        assert data["verdict"] == "ai"
        assert data["risk_level"] == "HIGH"
        assert data["cascade_path"] == "slow"
        assert data["inference_ms"] == 42.5
        assert data["warnings"] == ["LOW_CONFIDENCE"]
        assert data["explanation"]["summary"] == "Стиль шаблонный."
        assert data["explanation"]["top_markers"][0]["feature"] == "ttr"
        # Текст не должен возвращаться (privacy-by-design).
        assert "text" not in data
        # disclaimer всегда присутствует.
        assert data["disclaimer"]

    async def test_foreign_user_gets_404(self, client):
        # Алиса создаёт анализ, Боб пытается его прочитать.
        alice = await _register(client, email="alice@example.com")
        bob = await _register(client, email="bob@example.com")
        aid = await _make_analysis(alice["user"]["id"])

        bob_access = bob["tokens"]["access_token"]
        resp = await client.get(
            f"/me/analyses/{aid}", headers={"Authorization": f"Bearer {bob_access}"}
        )
        # Сознательно 404 (не 403): иначе утекает факт существования id.
        assert resp.status_code == 404

    async def test_anonymous_analysis_not_visible_to_user(self, client):
        # user_id=None (анонимный sync /analyze) не должен попадать в чей-то ЛК.
        body = await _register(client, email="claire@example.com")
        access = body["tokens"]["access_token"]
        aid = await _make_analysis(user_id=None)

        resp = await client.get(
            f"/me/analyses/{aid}", headers={"Authorization": f"Bearer {access}"}
        )
        assert resp.status_code == 404

    async def test_nonexistent_id_returns_404(self, client):
        body = await _register(client, email="dave@example.com")
        access = body["tokens"]["access_token"]
        fake_id = str(uuid7())

        resp = await client.get(
            f"/me/analyses/{fake_id}", headers={"Authorization": f"Bearer {access}"}
        )
        assert resp.status_code == 404

    async def test_invalid_uuid_returns_404(self, client):
        body = await _register(client, email="eve@example.com")
        access = body["tokens"]["access_token"]

        resp = await client.get(
            "/me/analyses/not-a-uuid", headers={"Authorization": f"Bearer {access}"}
        )
        # Те же 404, что для несуществующего — формат не утекает.
        assert resp.status_code == 404

    async def test_unauthenticated_returns_401(self, client):
        # Без bearer-токена — HTTPBearer auto_error=True → 401, до проверки id.
        resp = await client.get(f"/me/analyses/{uuid7()}")
        assert resp.status_code == 401

    async def test_explanation_absent_when_not_saved(self, client):
        body = await _register(client, email="frank@example.com")
        access = body["tokens"]["access_token"]
        user_id = body["user"]["id"]
        aid = await _make_analysis(user_id, explanation=None)

        resp = await client.get(
            f"/me/analyses/{aid}", headers={"Authorization": f"Bearer {access}"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["explanation"] is None
