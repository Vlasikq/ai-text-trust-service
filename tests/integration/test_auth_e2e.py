"""E2E на refresh-token rotation + reuse-detection."""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text as sql_text

from app.database import engine as engine_mod
from tests.conftest import make_test_app

pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def app_with_engine(postgres_url, _engine):
    engine_mod.init_engine(postgres_url, pool_size=2, max_overflow=2)
    yield make_test_app(routers=("auth",), db_enabled=True)
    await engine_mod.dispose_engine()


@pytest_asyncio.fixture(autouse=True, loop_scope="session")
async def _wipe(app_with_engine):
    async with engine_mod.async_session() as s:
        # FK refresh_tokens → users; truncate в правильном порядке.
        await s.execute(sql_text("DELETE FROM refresh_tokens"))
        await s.execute(sql_text("DELETE FROM users"))
        await s.commit()
    yield
    async with engine_mod.async_session() as s:
        await s.execute(sql_text("DELETE FROM refresh_tokens"))
        await s.execute(sql_text("DELETE FROM users"))
        await s.commit()


@pytest_asyncio.fixture(loop_scope="session")
async def client(app_with_engine):
    transport = ASGITransport(app=app_with_engine)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def test_refresh_token_reuse_revokes_family(client):
    # Регистрируем нового пользователя — получаем первую пару (T1).
    reg = await client.post(
        "/auth/register",
        json={"email": "alice@example.com", "password": "StrongPass1!"},
    )
    assert reg.status_code == 201, reg.text
    t1 = reg.json()["tokens"]["refresh_token"]

    # Ротация: T1 → T2. T1 после этого revoked.
    rot = await client.post("/auth/refresh", json={"refresh_token": t1})
    assert rot.status_code == 200, rot.text
    t2 = rot.json()["refresh_token"]
    assert t2 != t1

    # Reuse T1 — должен поймать reuse-detection и закрыть всю цепь.
    reuse = await client.post("/auth/refresh", json={"refresh_token": t1})
    assert reuse.status_code == 401

    # T2, выданный легитимно, теперь тоже revoked (revoke_all_for_user закрыл цепь).
    after_reuse = await client.post("/auth/refresh", json={"refresh_token": t2})
    assert after_reuse.status_code == 401
