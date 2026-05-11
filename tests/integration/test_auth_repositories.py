"""Тесты UserRepository + RefreshTokenRepository.

Покрытие:
  - User: create, get_by_id, get_by_email (case-insensitive),
    unique email constraint, update_last_login
  - RefreshToken: create, get_by_token_hash, revoke (с replaced_by),
    revoke_all_for_user
  - reuse-detection: revoke_all_for_user после presented revoked token
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text as sql_text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker
from uuid_utils import uuid7

from app.database.models import RefreshToken, User, UserRole, UserStatus
from app.database.repositories import RefreshTokenRepository, UserRepository

pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest.fixture(autouse=True)
async def _cleanup_auth(db_session):
    """users и refresh_tokens делятся между тестами — чистим до и после.

    refresh_tokens.user_id ON DELETE CASCADE → DELETE FROM users унесёт и токены.
    """
    await db_session.execute(sql_text("DELETE FROM refresh_tokens"))
    await db_session.execute(sql_text("DELETE FROM users"))
    await db_session.commit()
    yield
    await db_session.execute(sql_text("DELETE FROM refresh_tokens"))
    await db_session.execute(sql_text("DELETE FROM users"))
    await db_session.commit()


def _make_user(email: str = "alice@example.com", **overrides) -> User:
    defaults = {
        "id": uuid7(),
        "email": email,
        "password_hash": "$argon2id$v=19$m=65536,t=3,p=4$abc$def",  # фейк, формат argon2
        "status": UserStatus.ACTIVE,
        "role": UserRole.USER,
    }
    defaults.update(overrides)
    return User(**defaults)


def _make_token(user_id, *, token_hash: str = "f" * 64, days: int = 30) -> RefreshToken:
    return RefreshToken(
        id=uuid7(),
        user_id=user_id,
        token_hash=token_hash,
        expires_at=datetime.now(timezone.utc) + timedelta(days=days),
        revoked=False,
    )


# ── UserRepository ────────────────────────────────────────────


class TestUserRepository:
    async def test_create_and_get_by_id(self, db_session, _engine):
        repo = UserRepository(db_session)
        user = _make_user()
        saved = await repo.create(user)
        await db_session.commit()

        assert str(saved.id) == str(user.id)
        assert saved.status == UserStatus.ACTIVE
        assert saved.role == UserRole.USER

        # Round-trip через независимую сессию: подтверждаем, что commit реально
        # материализовался в Postgres и виден другому клиенту, а не только
        # сидит в SQLAlchemy Identity Map текущей сессии. Без этого тест
        # пропустит баг, при котором репозиторий молча не дотащил INSERT до БД.
        factory = async_sessionmaker(_engine, expire_on_commit=False)
        async with factory() as fresh_session:
            fresh_repo = UserRepository(fresh_session)
            found = await fresh_repo.get_by_id(user.id)

        assert found is not None
        assert found.email == "alice@example.com"
        assert found.status == UserStatus.ACTIVE
        assert found.role == UserRole.USER
        # created_at заполняется server-default 'now()' — проверяем, что round-trip
        # вернул timezone-aware datetime, а не None.
        assert found.created_at is not None
        assert found.created_at.tzinfo is not None
        # last_login_at не выставлялся при create — должен прийти как None из БД.
        assert found.last_login_at is None

    async def test_get_by_email_case_insensitive(self, db_session):
        repo = UserRepository(db_session)
        await repo.create(_make_user(email="Bob@Example.COM"))
        await db_session.commit()

        # Запрос в нижнем регистре с пробелами должен найти пользователя.
        found = await repo.get_by_email("  bob@example.com  ")
        assert found is not None
        assert found.email == "Bob@Example.COM"  # хранится как введён

    async def test_get_by_email_not_found(self, db_session):
        repo = UserRepository(db_session)
        result = await repo.get_by_email("nobody@example.com")
        assert result is None

    async def test_unique_email_constraint(self, db_session):
        repo = UserRepository(db_session)
        await repo.create(_make_user(email="duplicate@example.com"))
        await db_session.commit()

        # Пытаемся вставить второго с тем же email → IntegrityError.
        with pytest.raises(IntegrityError):
            await repo.create(_make_user(email="duplicate@example.com"))
            await db_session.commit()
        await db_session.rollback()

    async def test_unique_email_case_insensitive(self, db_session):
        """Регистр в email не должен открывать второй аккаунт на ту же почту."""
        repo = UserRepository(db_session)
        await repo.create(_make_user(email="Owner@Example.com"))
        await db_session.commit()

        with pytest.raises(IntegrityError):
            await repo.create(_make_user(email="owner@example.com"))
            await db_session.commit()
        await db_session.rollback()

    async def test_update_last_login(self, db_session):
        repo = UserRepository(db_session)
        user = await repo.create(_make_user())
        await db_session.commit()
        assert user.last_login_at is None

        await repo.update_last_login(user)
        await db_session.commit()
        await db_session.refresh(user)
        assert user.last_login_at is not None
        # Проверяем что время свежее (timezone-aware UTC).
        delta = datetime.now(timezone.utc) - user.last_login_at
        assert delta.total_seconds() < 5


# ── RefreshTokenRepository ────────────────────────────────────


class TestRefreshTokenRepository:
    async def test_create_and_get_by_hash(self, db_session):
        users = UserRepository(db_session)
        tokens = RefreshTokenRepository(db_session)

        user = await users.create(_make_user())
        await db_session.commit()

        token = await tokens.create(_make_token(user.id, token_hash="a" * 64))
        await db_session.commit()

        found = await tokens.get_by_token_hash("a" * 64)
        assert found is not None
        assert str(found.user_id) == str(user.id)
        assert found.revoked is False
        assert str(found.id) == str(token.id)

    async def test_get_by_hash_not_found(self, db_session):
        tokens = RefreshTokenRepository(db_session)
        result = await tokens.get_by_token_hash("z" * 64)
        assert result is None

    async def test_unique_token_hash(self, db_session):
        """Один и тот же hash дважды — UNIQUE-конфликт."""
        users = UserRepository(db_session)
        tokens = RefreshTokenRepository(db_session)
        user = await users.create(_make_user())
        await db_session.commit()

        await tokens.create(_make_token(user.id, token_hash="b" * 64))
        await db_session.commit()
        with pytest.raises(IntegrityError):
            await tokens.create(_make_token(user.id, token_hash="b" * 64))
            await db_session.commit()
        await db_session.rollback()

    async def test_revoke_with_rotation(self, db_session):
        """Refresh-rotation: старый revoked + replaced_by_id указывает на новый."""
        users = UserRepository(db_session)
        tokens = RefreshTokenRepository(db_session)
        user = await users.create(_make_user())
        await db_session.commit()

        old = await tokens.create(_make_token(user.id, token_hash="o" * 64))
        new = await tokens.create(_make_token(user.id, token_hash="n" * 64))
        await db_session.commit()

        await tokens.revoke(old, replaced_by=new)
        await db_session.commit()

        await db_session.refresh(old)
        assert old.revoked is True
        assert old.revoked_at is not None
        assert str(old.replaced_by_id) == str(new.id)

        # Новый — всё ещё активен.
        await db_session.refresh(new)
        assert new.revoked is False

    async def test_revoke_without_replacement(self, db_session):
        """Logout: revoke без replaced_by — конец цепи."""
        users = UserRepository(db_session)
        tokens = RefreshTokenRepository(db_session)
        user = await users.create(_make_user())
        await db_session.commit()

        token = await tokens.create(_make_token(user.id, token_hash="c" * 64))
        await db_session.commit()

        await tokens.revoke(token)
        await db_session.commit()
        await db_session.refresh(token)
        assert token.revoked is True
        assert token.replaced_by_id is None

    async def test_revoke_all_for_user_invalidates_chain(self, db_session):
        """Reuse-detection: revoke_all_for_user должен закрыть только активные токены.

        Сценарий: у пользователя 3 токена — два активных и один уже revoked. После
        revoke_all_for_user активные становятся revoked, а уже revoked не трогаем
        (revoked_at не перезаписывается).
        """
        users = UserRepository(db_session)
        tokens = RefreshTokenRepository(db_session)
        user = await users.create(_make_user())
        await db_session.commit()

        already_revoked = await tokens.create(_make_token(user.id, token_hash="r" * 64))
        active1 = await tokens.create(_make_token(user.id, token_hash="1" * 64))
        active2 = await tokens.create(_make_token(user.id, token_hash="2" * 64))
        await db_session.commit()

        # Помечаем один токен revoked заранее.
        original_revoked_at = datetime.now(timezone.utc) - timedelta(hours=1)
        already_revoked.revoked = True
        already_revoked.revoked_at = original_revoked_at
        await db_session.commit()

        n = await tokens.revoke_all_for_user(user.id)
        await db_session.commit()
        assert n == 2  # только два активных токена попали под revoke

        await db_session.refresh(active1)
        await db_session.refresh(active2)
        await db_session.refresh(already_revoked)

        assert active1.revoked is True and active1.revoked_at is not None
        assert active2.revoked is True and active2.revoked_at is not None
        # Старый revoked токен не перезаписан.
        assert already_revoked.revoked_at == original_revoked_at

    async def test_revoke_all_for_user_with_no_tokens(self, db_session):
        """Пользователь без активных токенов — revoke_all возвращает 0."""
        users = UserRepository(db_session)
        tokens = RefreshTokenRepository(db_session)
        user = await users.create(_make_user())
        await db_session.commit()

        n = await tokens.revoke_all_for_user(user.id)
        assert n == 0

    async def test_user_cascade_delete_removes_tokens(self, db_session):
        """ON DELETE CASCADE: удаление пользователя сносит его refresh-токены."""
        users = UserRepository(db_session)
        tokens = RefreshTokenRepository(db_session)
        user = await users.create(_make_user())
        await db_session.commit()

        await tokens.create(_make_token(user.id, token_hash="d" * 64))
        await tokens.create(_make_token(user.id, token_hash="e" * 64))
        await db_session.commit()

        await db_session.execute(
            sql_text(f"DELETE FROM users WHERE id = '{user.id}'::uuid")
        )
        await db_session.commit()

        # Все токены пользователя ушли.
        result = await db_session.execute(
            sql_text(f"SELECT COUNT(*) FROM refresh_tokens WHERE user_id = '{user.id}'::uuid")
        )
        assert result.scalar() == 0
