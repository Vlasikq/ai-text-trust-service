"""Тесты /auth/{register,login,refresh,logout} + /me + /me/analyses.

Подход: httpx.AsyncClient + ASGITransport — все запросы идут через тот же event loop,
что и наш AsyncEngine, поэтому asyncpg-соединения не падают на Windows ProactorLoop.
"""

from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text as sql_text
from uuid_utils import uuid7

from app.auth.tokens import hash_refresh_token
from app.database import engine as engine_mod
from app.database.models import Analysis
from app.database.uow import UnitOfWork
from tests.conftest import make_test_app

pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def app_with_engine(postgres_url, _engine):
    """Engine инициализируется в session loop, таблицы уже созданы фикстурой `_engine`."""
    engine_mod.init_engine(postgres_url, pool_size=2, max_overflow=2)
    yield make_test_app(routers=("auth", "me"))
    await engine_mod.dispose_engine()


@pytest_asyncio.fixture(autouse=True, loop_scope="session")
async def _wipe_auth_tables(app_with_engine):
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


# ── helpers ───────────────────────────────────────────────────


async def _register(
    client: AsyncClient, email: str = "alice@example.com", password: str = "supersecret123"
) -> dict:
    r = await client.post("/auth/register", json={"email": email, "password": password})
    return r.json()


# ── /auth/register ────────────────────────────────────────────


class TestRegister:
    async def test_creates_user_and_returns_tokens(self, client):
        resp = await client.post(
            "/auth/register",
            json={"email": "alice@example.com", "password": "supersecret123"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["user"]["email"] == "alice@example.com"
        assert body["user"]["role"] == "user"
        assert body["user"]["status"] == "active"
        assert body["tokens"]["token_type"] == "bearer"
        assert len(body["tokens"]["access_token"]) > 20
        assert len(body["tokens"]["refresh_token"]) > 20

    async def test_duplicate_email_returns_409(self, client):
        await client.post(
            "/auth/register",
            json={"email": "dup@example.com", "password": "supersecret123"},
        )
        resp = await client.post(
            "/auth/register",
            json={"email": "dup@example.com", "password": "anotherpass456"},
        )
        assert resp.status_code == 409

    async def test_invalid_email_rejected(self, client):
        resp = await client.post(
            "/auth/register",
            json={"email": "not-an-email", "password": "supersecret123"},
        )
        assert resp.status_code == 422

    async def test_short_password_rejected(self, client):
        resp = await client.post(
            "/auth/register",
            json={"email": "e@example.com", "password": "short"},
        )
        assert resp.status_code == 422


# ── /auth/login ───────────────────────────────────────────────


class TestLogin:
    async def test_login_returns_tokens(self, client):
        await _register(client, email="bob@example.com")
        resp = await client.post(
            "/auth/login",
            json={"email": "bob@example.com", "password": "supersecret123"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["access_token"]
        assert body["refresh_token"]

    async def test_wrong_password_returns_401(self, client):
        await _register(client, email="bob@example.com")
        resp = await client.post(
            "/auth/login",
            json={"email": "bob@example.com", "password": "WRONG-password-1"},
        )
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Invalid email or password"

    async def test_unknown_email_returns_401(self, client):
        resp = await client.post(
            "/auth/login",
            json={"email": "ghost@example.com", "password": "supersecret123"},
        )
        assert resp.status_code == 401

    async def test_login_case_insensitive_email(self, client):
        await _register(client, email="Charlie@Example.com")
        resp = await client.post(
            "/auth/login",
            json={"email": "CHARLIE@example.com", "password": "supersecret123"},
        )
        assert resp.status_code == 200


# ── /auth/refresh ─────────────────────────────────────────────


class TestRefresh:
    async def test_rotation_returns_new_pair_and_revokes_old(self, client):
        body = await _register(client, email="dave@example.com")
        old = body["tokens"]["refresh_token"]
        resp = await client.post("/auth/refresh", json={"refresh_token": old})
        assert resp.status_code == 200
        new_refresh = resp.json()["refresh_token"]
        assert new_refresh != old

        # Старый refresh теперь revoked → reuse-detection.
        reuse = await client.post("/auth/refresh", json={"refresh_token": old})
        assert reuse.status_code == 401

    async def test_reuse_revokes_entire_chain(self, client):
        body = await _register(client, email="evelyn@example.com")
        old = body["tokens"]["refresh_token"]
        # Цепь refresh-rotation: old -> new1 -> new2 (все активные кроме old и new1).
        new1 = (await client.post("/auth/refresh", json={"refresh_token": old})).json()[
            "refresh_token"
        ]
        new2 = (await client.post("/auth/refresh", json={"refresh_token": new1})).json()[
            "refresh_token"
        ]
        # new2 ещё рабочий.
        check = await client.post("/auth/refresh", json={"refresh_token": new2})
        assert check.status_code == 200

        # Атакующий предъявляет revoked old → reuse-detection ломает всю цепь.
        attack = await client.post("/auth/refresh", json={"refresh_token": old})
        assert attack.status_code == 401

        async with engine_mod.async_session() as s:
            r = await s.execute(
                sql_text("SELECT COUNT(*) FROM refresh_tokens WHERE revoked = false")
            )
            assert r.scalar() == 0

    async def test_refresh_unknown_token_returns_401(self, client):
        resp = await client.post(
            "/auth/refresh", json={"refresh_token": "nonexistent-refresh-token"}
        )
        assert resp.status_code == 401

    async def test_expired_refresh_rejected(self, client):
        body = await _register(client, email="frank@example.com")
        plain = body["tokens"]["refresh_token"]
        h = hash_refresh_token(plain)
        past = datetime.now(timezone.utc) - timedelta(seconds=1)

        async with engine_mod.async_session() as s:
            await s.execute(
                sql_text(
                    "UPDATE refresh_tokens SET expires_at = :exp WHERE token_hash = :h"
                ).bindparams(exp=past, h=h)
            )
            await s.commit()

        resp = await client.post("/auth/refresh", json={"refresh_token": plain})
        assert resp.status_code == 401


# ── /auth/logout ──────────────────────────────────────────────


class TestLogout:
    async def test_logout_revokes_refresh_token(self, client):
        body = await _register(client, email="hank@example.com")
        refresh = body["tokens"]["refresh_token"]

        resp = await client.post("/auth/logout", json={"refresh_token": refresh})
        assert resp.status_code == 204

        retry = await client.post("/auth/refresh", json={"refresh_token": refresh})
        assert retry.status_code == 401

    async def test_logout_idempotent(self, client):
        body = await _register(client, email="ida@example.com")
        refresh = body["tokens"]["refresh_token"]
        # Дважды разлогиниваемся — без ошибок.
        assert (await client.post("/auth/logout", json={"refresh_token": refresh})).status_code == 204
        assert (await client.post("/auth/logout", json={"refresh_token": refresh})).status_code == 204

    async def test_logout_unknown_token_idempotent(self, client):
        resp = await client.post("/auth/logout", json={"refresh_token": "garbage-refresh-string"})
        assert resp.status_code == 204


# ── /me + /me/analyses (list) ─────────────────────────────────
# Эти тесты живут здесь, а не в test_me_detail.py / test_me_stats.py:
# проверяют связку auth-flow + "после логина читаю свою историю".
# Detail (GET /me/analyses/{id}) и stats (/me/stats) выделены в отдельные
# файлы — у них своя бизнес-логика (owner-check, временные окна).


class TestMeEndpoints:
    async def test_me_requires_auth(self, client):
        resp = await client.get("/me")
        # FastAPI HTTPBearer auto_error=True → 401 без креденшелов.
        assert resp.status_code == 401

    async def test_me_returns_user(self, client):
        body = await _register(client, email="jack@example.com")
        access = body["tokens"]["access_token"]

        resp = await client.get("/me", headers={"Authorization": f"Bearer {access}"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["email"] == "jack@example.com"
        assert body["role"] == "user"

    async def test_me_invalid_token_returns_401(self, client):
        resp = await client.get(
            "/me", headers={"Authorization": "Bearer not-a-valid-jwt"}
        )
        assert resp.status_code == 401

    async def test_my_analyses_empty(self, client):
        body = await _register(client, email="kate@example.com")
        access = body["tokens"]["access_token"]

        resp = await client.get(
            "/me/analyses", headers={"Authorization": f"Bearer {access}"}
        )
        assert resp.status_code == 200
        assert resp.json() == {"items": [], "next_cursor": None}

    async def test_my_analyses_filters_by_user(self, client):
        r1 = await _register(client, email="leo@example.com")
        r2 = await _register(client, email="mia@example.com")
        leo_id = r1["user"]["id"]
        mia_id = r2["user"]["id"]
        leo_access = r1["tokens"]["access_token"]

        async with UnitOfWork() as uow:
            for i in range(2):
                await uow.analyses.save(
                    Analysis(
                        id=uuid7(),
                        request_id=uuid7(),
                        text_hash=f"{i}" * 64,
                        text_length=200,
                        status="SUCCESS",
                        verdict="ai",
                        confidence=0.9,
                        risk_level="HIGH",
                        detector_used="cascade",
                        cascade_path="fast",
                        inference_ms=10.0,
                        cached=False,
                        model_version="v1",
                        service_version="0.1.0",
                        user_id=leo_id,
                        warnings=[],
                        requested_at=datetime.now(timezone.utc) - timedelta(minutes=i),
                    )
                )
            await uow.analyses.save(
                Analysis(
                    id=uuid7(),
                    request_id=uuid7(),
                    text_hash="m" * 64,
                    text_length=200,
                    status="SUCCESS",
                    verdict="human",
                    confidence=0.2,
                    risk_level="LOW",
                    detector_used="cascade",
                    cascade_path="fast",
                    inference_ms=10.0,
                    cached=False,
                    model_version="v1",
                    service_version="0.1.0",
                    user_id=mia_id,
                    warnings=[],
                    requested_at=datetime.now(timezone.utc),
                )
            )
            await uow.analyses.save(
                Analysis(
                    id=uuid7(),
                    request_id=uuid7(),
                    text_hash="z" * 64,
                    text_length=200,
                    status="SUCCESS",
                    verdict="ai",
                    confidence=0.7,
                    risk_level="MEDIUM",
                    detector_used="tfidf",
                    cascade_path=None,
                    inference_ms=5.0,
                    cached=False,
                    model_version="v1",
                    service_version="0.1.0",
                    user_id=None,
                    warnings=[],
                    requested_at=datetime.now(timezone.utc),
                )
            )
            await uow.commit()

        resp = await client.get(
            "/me/analyses", headers={"Authorization": f"Bearer {leo_access}"}
        )
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert len(items) == 2
        for it in items:
            assert it["verdict"] == "ai"

    async def test_my_analyses_pagination(self, client):
        body = await _register(client, email="nico@example.com")
        user_id = body["user"]["id"]
        access = body["tokens"]["access_token"]

        async with UnitOfWork() as uow:
            for i in range(5):
                await uow.analyses.save(
                    Analysis(
                        id=uuid7(),
                        request_id=uuid7(),
                        text_hash=f"{i:064d}",
                        text_length=200,
                        status="SUCCESS",
                        verdict="ai",
                        confidence=0.9,
                        risk_level="HIGH",
                        detector_used="cascade",
                        cascade_path="fast",
                        inference_ms=10.0,
                        cached=False,
                        model_version="v1",
                        service_version="0.1.0",
                        user_id=user_id,
                        warnings=[],
                        requested_at=datetime.now(timezone.utc) - timedelta(minutes=i),
                    )
                )
            await uow.commit()

        page1 = (
            await client.get(
                "/me/analyses?limit=2",
                headers={"Authorization": f"Bearer {access}"},
            )
        ).json()
        assert len(page1["items"]) == 2
        assert page1["next_cursor"] is not None

        page2 = (
            await client.get(
                "/me/analyses",
                params={"limit": 2, "cursor": page1["next_cursor"]},
                headers={"Authorization": f"Bearer {access}"},
            )
        ).json()
        assert len(page2["items"]) == 2
        ids1 = {it["id"] for it in page1["items"]}
        ids2 = {it["id"] for it in page2["items"]}
        assert ids1.isdisjoint(ids2)

    async def test_disabled_user_cannot_use_token(self, client):
        body = await _register(client, email="olga@example.com")
        access = body["tokens"]["access_token"]

        async with UnitOfWork() as uow:
            user = await uow.users.get_by_id(body["user"]["id"])
            user.status = "disabled"
            await uow.commit()

        resp = await client.get("/me", headers={"Authorization": f"Bearer {access}"})
        assert resp.status_code == 401
