"""Тесты GET /me/stats — персональная статистика для дашборда.

Контракт:
  - Empty state (нет анализов) → 200, total_scans=0, daily содержит окно нулей.
  - 7d / 30d / 90d / all периоды.
  - Daily zero-padded: каждая дата в окне присутствует, count=0 для пустых.
  - Чужие анализы (user_id != current) не попадают в агрегаты.
  - Запросы вне окна (старше period_start) фильтруются.
  - Без auth → 401.
"""

from datetime import datetime, timedelta, timezone

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


async def _register(client: AsyncClient, email: str = "stats@example.com") -> dict:
    r = await client.post(
        "/auth/register", json={"email": email, "password": "supersecret123"}
    )
    return r.json()


def _spec(
    days_ago: int,
    verdict: str | None = None,
    risk_level: str | None = None,
    detector: str = "tfidf",
    confidence: float = 0.7,
) -> dict:
    return {
        "days_ago": days_ago,
        "verdict": verdict,
        "risk_level": risk_level,
        "detector": detector,
        "confidence": confidence,
    }


async def _make_analyses(
    user_id: str | None,
    specs: list[dict],
) -> None:
    """Создать анализы из spec'ов; ключи: days_ago, verdict, risk_level, detector, confidence."""
    async with UnitOfWork() as uow:
        for spec in specs:
            ts = datetime.now(timezone.utc) - timedelta(days=spec["days_ago"])
            await uow.analyses.save(
                Analysis(
                    id=uuid7(),
                    request_id=uuid7(),
                    text_hash="h" * 64,
                    text_length=400,
                    status="SUCCESS",
                    verdict=spec.get("verdict"),
                    confidence=spec.get("confidence", 0.7),
                    risk_level=spec.get("risk_level"),
                    detector_used=spec.get("detector", "tfidf"),
                    cascade_path=None,
                    inference_ms=10.0,
                    cached=False,
                    model_version="v1",
                    service_version="0.1.0",
                    user_id=user_id,
                    warnings=[],
                    requested_at=ts,
                )
            )
        await uow.commit()


class TestStatsEndpoint:
    async def test_empty_state(self, client):
        body = await _register(client, email="empty@example.com")
        access = body["tokens"]["access_token"]

        resp = await client.get(
            "/me/stats", headers={"Authorization": f"Bearer {access}"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["period"] == "30d"
        assert data["total_scans"] == 0
        assert data["by_verdict"] == {}
        assert data["by_risk_level"] == {}
        assert data["avg_confidence"] is None
        # 30 дней + сегодня = 31 bucket, все нулевые.
        assert len(data["daily"]) == 31
        assert all(b["total"] == 0 for b in data["daily"])

    async def test_aggregates_within_period(self, client):
        body = await _register(client, email="agg@example.com")
        access = body["tokens"]["access_token"]
        user_id = body["user"]["id"]

        await _make_analyses(
            user_id,
            [
                _spec(0, "ai", "HIGH", "cascade", 0.9),
                _spec(1, "ai", "HIGH", "tfidf", 0.8),
                _spec(2, "human", "LOW", "tfidf", 0.1),
                _spec(5, "ai", "MEDIUM", "cascade", 0.6),
            ],
        )

        resp = await client.get(
            "/me/stats?period=7d", headers={"Authorization": f"Bearer {access}"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["period"] == "7d"
        assert data["total_scans"] == 4
        assert data["by_verdict"] == {"ai": 3, "human": 1}
        assert data["by_risk_level"] == {"HIGH": 2, "LOW": 1, "MEDIUM": 1}
        assert data["by_detector"] == {"tfidf": 2, "cascade": 2}
        assert data["avg_confidence"] is not None
        assert abs(data["avg_confidence"] - (0.9 + 0.8 + 0.1 + 0.6) / 4) < 1e-6

    async def test_period_filters_old_records(self, client):
        body = await _register(client, email="filter@example.com")
        access = body["tokens"]["access_token"]
        user_id = body["user"]["id"]

        # 1 запись внутри 7d, 1 запись 30 дней назад → в 7d не попадает.
        await _make_analyses(
            user_id,
            [
                _spec(1, "ai", "HIGH", "tfidf", 0.9),
                _spec(30, "human", "LOW", "tfidf", 0.1),
            ],
        )

        r7 = await client.get(
            "/me/stats?period=7d", headers={"Authorization": f"Bearer {access}"}
        )
        assert r7.json()["total_scans"] == 1

        r90 = await client.get(
            "/me/stats?period=90d", headers={"Authorization": f"Bearer {access}"}
        )
        assert r90.json()["total_scans"] == 2

    async def test_daily_zero_padding(self, client):
        body = await _register(client, email="pad@example.com")
        access = body["tokens"]["access_token"]
        user_id = body["user"]["id"]

        # Только 2 записи: сегодня и 3 дня назад. Между ними должны быть 0-buckets.
        await _make_analyses(
            user_id,
            [
                {"days_ago": 0, "verdict": "ai", "risk_level": "HIGH"},
                {"days_ago": 3, "verdict": "ai", "risk_level": "HIGH"},
            ],
        )

        resp = await client.get(
            "/me/stats?period=7d", headers={"Authorization": f"Bearer {access}"}
        )
        data = resp.json()
        # 7 дней назад до сегодня = 8 buckets (включительно).
        assert len(data["daily"]) == 8
        # Каждый bucket имеет дату и total.
        totals = [b["total"] for b in data["daily"]]
        assert sum(totals) == 2  # суммарно 2 записи
        # Найти bucket с total=2 не должно быть, два разных дня.
        assert max(totals) == 1
        # Дни в порядке возрастания.
        dates = [b["date"] for b in data["daily"]]
        assert dates == sorted(dates)

    async def test_isolation_between_users(self, client):
        alice = await _register(client, email="alice-stats@example.com")
        bob = await _register(client, email="bob-stats@example.com")
        await _make_analyses(
            alice["user"]["id"],
            [{"days_ago": 0, "verdict": "ai", "risk_level": "HIGH"}],
        )
        await _make_analyses(
            bob["user"]["id"],
            [
                {"days_ago": 0, "verdict": "human", "risk_level": "LOW"},
                {"days_ago": 1, "verdict": "human", "risk_level": "LOW"},
            ],
        )

        alice_access = alice["tokens"]["access_token"]
        resp = await client.get(
            "/me/stats", headers={"Authorization": f"Bearer {alice_access}"}
        )
        data = resp.json()
        assert data["total_scans"] == 1
        assert data["by_verdict"] == {"ai": 1}

    async def test_anonymous_analyses_excluded(self, client):
        body = await _register(client, email="anon@example.com")
        access = body["tokens"]["access_token"]
        user_id = body["user"]["id"]

        await _make_analyses(
            user_id,
            [{"days_ago": 0, "verdict": "ai", "risk_level": "HIGH"}],
        )
        # Анонимный анализ (user_id=None) не должен попасть в чьи-либо stats.
        await _make_analyses(
            None, [{"days_ago": 0, "verdict": "human", "risk_level": "LOW"}]
        )

        resp = await client.get(
            "/me/stats", headers={"Authorization": f"Bearer {access}"}
        )
        data = resp.json()
        assert data["total_scans"] == 1
        assert data["by_verdict"] == {"ai": 1}

    async def test_period_all_uses_earliest(self, client):
        body = await _register(client, email="all@example.com")
        access = body["tokens"]["access_token"]
        user_id = body["user"]["id"]

        await _make_analyses(
            user_id,
            [
                {"days_ago": 100, "verdict": "ai", "risk_level": "HIGH"},
                {"days_ago": 0, "verdict": "human", "risk_level": "LOW"},
            ],
        )

        resp = await client.get(
            "/me/stats?period=all", headers={"Authorization": f"Bearer {access}"}
        )
        data = resp.json()
        assert data["period"] == "all"
        assert data["total_scans"] == 2
        # Daily покрывает 100 дней + сегодня = 101 bucket.
        assert len(data["daily"]) >= 100

    async def test_invalid_period_rejected(self, client):
        body = await _register(client, email="bad@example.com")
        access = body["tokens"]["access_token"]

        resp = await client.get(
            "/me/stats?period=42d", headers={"Authorization": f"Bearer {access}"}
        )
        # FastAPI валидация Enum → 422.
        assert resp.status_code == 422

    async def test_unauthenticated(self, client):
        resp = await client.get("/me/stats")
        assert resp.status_code == 401
