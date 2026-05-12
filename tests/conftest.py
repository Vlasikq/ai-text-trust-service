"""
Shared test fixtures + утилиты для построения тестового FastAPI-приложения.

DB-фикстуры:
  - postgres_url: TEST_DATABASE_URL → существующий Postgres ИЛИ testcontainers ИЛИ skip.
  - _engine: один engine на сессию, метадата создаётся/чистится автоматически.
  - db_session: одна сессия на тест, rollback на teardown.

Хелперы (импортируются из тестовых модулей напрямую):
  - FakeDetector / FakeExplainer — параметризованные test doubles.
  - make_test_app(routers=...) — фабрика FastAPI для интеграционных тестов.

Также глобально отключаем slowapi rate-limit на время всей тестовой сессии:
multi-test регистрации/логины забивают лимиты `5/hour`, и мы не хотим
тестировать сам limiter тут — это делается в отдельном test_middleware.py
с явным enable.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncGenerator, Iterable
from itertools import cycle
from typing import Any, Iterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.cache import DetectionCache
from app.config import Settings
from app.database.models import Base
from app.detectors.base import BaseDetector, DetectionResult
from app.middleware.ratelimit import limiter
from app.preprocessing.pipeline import TextPreprocessor
from app.schemas import Explanation, StyleMarker

# Отключаем сразу при импорте — до сборки приложений в make_test_app().
# Тесты, которым нужен включённый limiter, ставят `limiter.enabled = True` локально.
limiter.enabled = False


# Дефолт Settings.is_production=True (fail-closed) — для тестов перекрываем
# на dev. Тесты прод-инвариантов сами monkeypatch'ат "true" и откатывают.
os.environ["IS_PRODUCTION"] = "false"


# ── Detect DB availability ────────────────────────────────────


def _get_postgres_url() -> str | None:
    url = os.getenv("TEST_DATABASE_URL")
    if url:
        return url
    try:
        import docker

        docker.from_env().ping()
        return None  # use testcontainers
    except Exception:
        return "SKIP"


_pg_result = _get_postgres_url()
_skip_db = _pg_result == "SKIP"
_use_existing_db = _pg_result is not None and _pg_result != "SKIP"


# ── Postgres URL ──────────────────────────────────────────────


@pytest.fixture(scope="session")
def postgres_url():
    if _skip_db:
        pytest.skip("No Docker / TEST_DATABASE_URL — skipping DB tests")

    if _use_existing_db:
        yield _pg_result
        return

    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        sync_url = pg.get_connection_url()
        yield sync_url.replace("psycopg2", "asyncpg")


# ── Engine ────────────────────────────────────────────────────


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def _engine(postgres_url):
    engine = create_async_engine(postgres_url, echo=False)
    if not _use_existing_db:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    yield engine
    if not _use_existing_db:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


# ── DB Session ────────────────────────────────────────────────


@pytest_asyncio.fixture(loop_scope="session")
async def db_session(_engine) -> AsyncGenerator[AsyncSession, None]:
    """Each test gets a fresh session. Rollback on teardown."""
    factory = async_sessionmaker(_engine, expire_on_commit=False)
    session = factory()
    try:
        yield session
    finally:
        await session.rollback()
        await session.close()


# ────────────────────────────────────────────────────────────────────
# Test doubles
# ────────────────────────────────────────────────────────────────────


class FakeDetector(BaseDetector):
    """Параметризованный stub-детектор. Покрывает все use-кейсы тестов:

      FakeDetector()                            — prob_ai=0.85, ready
      FakeDetector(prob_ai=0.10)                — low-confidence (verdict=human)
      FakeDetector(ready=False)                 — для /ready 503-кейса
      FakeDetector(sleep_s=0.5)                 — для timeout-веток
      FakeDetector(raises=ValueError("boom"))   — для error-веток
      FakeDetector(metadata={"cascade_path":"slow"}) — проверка проброса metadata
      FakeDetector(prob_seq=[0.1, 0.9])         — чередование (segment_scoring)
    """

    def __init__(
        self,
        prob_ai: float = 0.85,
        *,
        method: str = "tfidf",
        ready: bool = True,
        sleep_s: float = 0.0,
        raises: Exception | None = None,
        metadata: dict | None = None,
        prob_seq: Iterable[float] | None = None,
        inference_ms: float = 5.0,
    ):
        self._prob_ai = prob_ai
        self._method = method
        self._ready = ready
        self._sleep_s = sleep_s
        self._raises = raises
        self._metadata = metadata or {}
        self._inference_ms = inference_ms
        self._prob_iter: Iterator[float] | None = (
            cycle(prob_seq) if prob_seq is not None else None
        )

    def load(self) -> None:  # pragma: no cover — нет артефактов
        pass

    def predict(self, text: str) -> DetectionResult:
        if self._sleep_s:
            time.sleep(self._sleep_s)
        if self._raises is not None:
            raise self._raises
        prob = next(self._prob_iter) if self._prob_iter is not None else self._prob_ai
        return DetectionResult(
            prob_ai=prob,
            method=self._method,
            inference_ms=self._inference_ms,
            metadata=dict(self._metadata),
        )

    def is_ready(self) -> bool:
        return self._ready


class FakeExplainer:
    """Stub-объяснитель: возвращает одиночный ttr-маркер.

      FakeExplainer()                       — дефолтное объяснение
      FakeExplainer(raises=Exception("boom")) — для error-веток
    """

    def __init__(self, *, raises: Exception | None = None) -> None:
        self._raises = raises

    def explain(self, text: str, top_n: int = 5) -> Explanation:
        if self._raises is not None:
            raise self._raises
        return Explanation(
            top_markers=[
                StyleMarker(
                    feature="ttr",
                    value=0.5,
                    human_baseline=0.4,
                    deviation_percent=25.0,
                    direction="above",
                )
            ],
            summary="Тестовое объяснение.",
        )


# ────────────────────────────────────────────────────────────────────
# Фабрика тестового FastAPI-приложения
# ────────────────────────────────────────────────────────────────────


def _resolve_router(name: str):
    """Lazy-импорт роутеров — избегаем circular на этапе сборки conftest."""
    if name == "analyze":
        from app.routers.analyze import router
        return router
    if name == "auth":
        from app.routers.auth import router
        return router
    if name == "batch":
        from app.routers.batch import router
        return router
    if name == "batch_me":
        from app.routers.batch import me_router
        return me_router
    if name == "extract":
        from app.routers.extract import router
        return router
    if name == "feedback":
        from app.routers.feedback import router
        return router
    if name == "health":
        from app.routers.health import router
        return router
    if name == "jobs":
        from app.routers.jobs import router
        return router
    if name == "me":
        from app.routers.me import router
        return router
    raise ValueError(f"unknown router: {name!r}")


def make_test_app(
    *,
    routers: Iterable[str] = ("analyze",),
    detector: BaseDetector | None = None,
    explainer: Any | None = None,
    calibrator: Any | None = None,
    db_enabled: bool = False,
    with_middleware: bool = False,
    cache_maxsize: int = 1024,
    **settings_kw: Any,
):
    """Универсальная фабрика тестового FastAPI-приложения.

    По умолчанию: только /analyze, FakeDetector(0.85), без БД, без middleware.
    Все параметры опциональны.

    Параметры:
      routers          — список имён: "analyze", "auth", "batch", "batch_me",
                         "extract", "feedback", "health", "jobs", "me".
      detector         — BaseDetector; default FakeDetector(prob_ai=0.85).
      explainer        — стаб с .explain(text); default None.
      calibrator       — стаб с .calibrate(prob, method); default None.
      db_enabled       — флаг app.state.db_enabled (нужен endpoint'ам с проверкой
                         доступности БД, например /feedback и /api/v1/batch).
      with_middleware  — добавить RequestLoggingMiddleware (correlation_id).
      cache_maxsize    — размер LRU-кэша DetectionCache.
      **settings_kw    — переопределения Settings (дефолт min_chars=10).
    """
    from fastapi import FastAPI

    app = FastAPI()
    app.state.limiter = limiter

    for name in routers:
        app.include_router(_resolve_router(name))

    settings_kw.setdefault("min_chars", 10)
    settings = Settings(**settings_kw)
    app.state.settings = settings
    app.state.preprocessor = TextPreprocessor(settings)
    app.state.detector = detector if detector is not None else FakeDetector()
    app.state.explainer = explainer
    app.state.calibrator = calibrator
    app.state.cache = DetectionCache(maxsize=cache_maxsize)
    app.state.db_enabled = db_enabled

    if with_middleware:
        from app.middleware.metrics import RequestLoggingMiddleware
        app.add_middleware(RequestLoggingMiddleware)

    return app
