"""Тесты get_current_user поверх реального /me-роутера.

Зачем integration, а не unit: dependency дёргает UnitOfWork → нужна реальная
сессия. Используем тот же подход, что в test_feedback.py — session-scoped engine
+ wipe фикстура.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text as sql_text

from app.auth.passwords import hash_password
from app.auth.tokens import create_access_token
from app.database import engine as engine_mod
from app.database.models import User, UserRole, UserStatus
from tests.conftest import make_test_app

pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def app_with_engine(postgres_url, _engine):
    engine_mod.init_engine(postgres_url, pool_size=2, max_overflow=2)
    yield make_test_app(routers=("me",), db_enabled=True)
    await engine_mod.dispose_engine()


@pytest_asyncio.fixture(autouse=True, loop_scope="session")
async def _wipe_users(app_with_engine):
    async with engine_mod.async_session() as s:
        await s.execute(sql_text("DELETE FROM users"))
        await s.commit()
    yield
    async with engine_mod.async_session() as s:
        await s.execute(sql_text("DELETE FROM users"))
        await s.commit()


@pytest_asyncio.fixture(loop_scope="session")
async def client(app_with_engine):
    transport = ASGITransport(app=app_with_engine)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def _create_user(*, status: str = UserStatus.ACTIVE) -> User:
    user = User(
        id=uuid.uuid4(),
        email=f"{uuid.uuid4().hex[:8]}@example.com",
        password_hash=hash_password("password123!"),
        status=status,
        role=UserRole.USER,
    )
    async with engine_mod.async_session() as s:
        s.add(user)
        await s.commit()
        await s.refresh(user)
    return user


def _token_for(app, user: User, *, ttl_minutes: int = 15) -> str:
    s = app.state.settings
    return create_access_token(
        user_id=str(user.id),
        role=user.role,
        secret=s.jwt_secret,
        issuer=s.jwt_issuer,
        audience=s.jwt_audience,
        algorithm=s.jwt_algorithm,
        ttl_minutes=ttl_minutes,
    )


class TestGetCurrentUser:
    async def test_valid_token_returns_user(self, client, app_with_engine):
        user = await _create_user()
        token = _token_for(app_with_engine, user)
        resp = await client.get("/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["email"] == user.email

    async def test_expired_token_returns_401(self, client, app_with_engine):
        user = await _create_user()
        # ttl=-1 минута → токен уже истёк.
        token = _token_for(app_with_engine, user, ttl_minutes=-1)
        resp = await client.get("/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    async def test_disabled_user_returns_401(self, client, app_with_engine):
        user = await _create_user(status=UserStatus.DISABLED)
        token = _token_for(app_with_engine, user)
        resp = await client.get("/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    async def test_missing_token_returns_401(self, client):
        resp = await client.get("/me")
        assert resp.status_code == 401
