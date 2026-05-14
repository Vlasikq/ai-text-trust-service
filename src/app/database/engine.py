"""
Async SQLAlchemy engine and session factory.

Usage:
    from app.database.engine import init_engine, async_session, dispose_engine

    # In lifespan:
    init_engine(settings)
    ...
    await dispose_engine()
"""

import logging

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

log = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
async_session: async_sessionmaker[AsyncSession] | None = None


def init_engine(
    database_url: str,
    pool_size: int = 5,
    max_overflow: int = 5,
) -> AsyncEngine:
    """Создаёт глобальный async-engine и фабрику сессий."""
    global _engine, async_session

    _engine = create_async_engine(
        database_url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_recycle=3600,
        echo=False,
    )
    async_session = async_sessionmaker(_engine, expire_on_commit=False)
    log.info("Database engine created: %s", database_url.split("@")[-1])
    return _engine


def get_engine() -> AsyncEngine:
    """Возвращает текущий engine (должен быть инициализирован init_engine)."""
    if _engine is None:
        raise RuntimeError("Database engine not initialized. Call init_engine() first.")
    return _engine


async def dispose_engine() -> None:
    """Закрывает engine на shutdown lifespan."""
    global _engine, async_session
    if _engine is not None:
        await _engine.dispose()
        log.info("Database engine disposed")
        _engine = None
        async_session = None
