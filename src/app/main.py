"""Фабрика FastAPI-приложения и lifespan.

В lifespan последовательно поднимаются детектор, калибратор, объяснятор,
кеш и подключение к БД, после чего идёт прогрев модели.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.cache import DetectionCache
from app.calibration.platt import ProbabilityCalibrator
from app.config import DetectorType, Settings, get_settings
from app.detectors.cascade import CascadeDetector
from app.detectors.tfidf import TFIDFDetector
from app.detectors.transformer import TransformerDetector
from app.explanation.markers import StyleExplainer
from app.logging_config import setup_logging
from app.middleware.metrics import RequestLoggingMiddleware, metrics_endpoint
from app.middleware.ratelimit import limiter
from app.preprocessing.pipeline import TextPreprocessor
from app.routers.analyze import router as analyze_router
from app.routers.auth import router as auth_router
from app.routers.batch import me_router as batch_me_router
from app.routers.batch import router as batch_router
from app.routers.extract import router as extract_router
from app.routers.feedback import router as feedback_router
from app.routers.health import router as health_router
from app.routers.jobs import router as jobs_router
from app.routers.me import router as me_router

log = logging.getLogger(__name__)


def _build_detector(settings: Settings):
    tfidf = TFIDFDetector(settings.artifacts_dir)

    if settings.detector_type == DetectorType.tfidf:
        tfidf.load()
        return tfidf

    if not settings.transformer_dir.exists():
        log.warning("Transformer dir not found: %s", settings.transformer_dir)
        if settings.detector_type == DetectorType.transformer:
            raise FileNotFoundError(
                f"Transformer required but not found: {settings.transformer_dir}"
            )
        # Каскад без трансформера: возвращаем TF-IDF без медленного пути.
        tfidf.load()
        cascade = CascadeDetector(fast=tfidf, slow=None)
        return cascade

    transformer = TransformerDetector(settings.transformer_dir)

    if settings.detector_type == DetectorType.transformer:
        transformer.load()
        return transformer

    cascade = CascadeDetector(
        fast=tfidf,
        slow=transformer,
        cascade_lo=settings.cascade_lo,
        cascade_hi=settings.cascade_hi,
    )
    cascade.load()
    return cascade


async def _init_database(app: FastAPI, settings: Settings) -> None:
    from app.database.engine import init_engine
    from app.database.models import ModelDeployment
    from app.database.uow import UnitOfWork

    try:
        init_engine(
            settings.database_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_pool_overflow,
        )
        app.state.db_enabled = True
        log.info("Database connected")

        # Record deployment
        async with UnitOfWork() as uow:
            deployment = ModelDeployment(
                model_version=settings.model_version,
                service_version=settings.service_version,
                detector_type=settings.detector_type.value,
                config_snapshot={
                    "cascade_lo": settings.cascade_lo,
                    "cascade_hi": settings.cascade_hi,
                    "min_chars": settings.min_chars,
                    "max_chars": settings.max_chars,
                },
            )
            await uow.deployments.save(deployment)
            await uow.commit()
            app.state.deployment_id = str(deployment.id)
            log.info("Deployment recorded: %s", deployment.id)

    except Exception:
        app.state.db_enabled = False
        app.state.deployment_id = None
        log.warning("Database not available — persistence disabled", exc_info=True)


_DEFAULT_JWT_SECRET_PREFIX = "dev-only-secret"


def _validate_settings_for_prod(settings: Settings) -> None:
    """Отказывает в старте, если в проде остались дефолтные секреты.

    Так отлавливается частая ошибка «забыл переопределить JWT_SECRET через ENV».
    Без этой проверки access-токены легко подделать, потому что дефолт открыт в коде.
    На локальной разработке (`is_production=false`) проверка пропускается.
    """
    if not settings.is_production:
        return
    if settings.jwt_secret.startswith(_DEFAULT_JWT_SECRET_PREFIX):
        raise RuntimeError(
            "JWT_SECRET не переопределён в production-окружении — "
            "отказ старта. Задайте ENV `JWT_SECRET=<random 32+ bytes>`."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    setup_logging(settings.log_level)
    _validate_settings_for_prod(settings)

    log.info("Starting AI Text Trust Service v%s", settings.service_version)

    # Складываем компоненты в app.state — отсюда их берут зависимости роутеров.
    app.state.settings = settings
    app.state.preprocessor = TextPreprocessor(settings)
    app.state.detector = _build_detector(settings)

    app.state.cache = DetectionCache(maxsize=settings.cache_maxsize)

    calibration_dir = settings.artifacts_dir / "calibration"
    calibrator = ProbabilityCalibrator(calibration_dir)
    try:
        calibrator.load()
        app.state.calibrator = calibrator
        log.info("Calibrator: %s", calibrator.get_diagnostics())
    except Exception:
        app.state.calibrator = None
        log.warning("Calibrator not loaded — raw probabilities will be used")

    baselines_path = settings.artifacts_dir / "human_baselines.json"
    explainer = StyleExplainer(baselines_path)
    try:
        explainer.load()
        app.state.explainer = explainer if explainer.is_ready() else None
    except Exception:
        app.state.explainer = None
        log.warning("Explainer not loaded — explanations will be unavailable")

    await _init_database(app, settings)

    # Один холостой инференс стабилизирует p95.
    try:
        app.state.detector.predict("Тестовый текст для прогрева модели.")
        log.info("Warmup complete")
    except Exception:
        log.warning("Warmup failed", exc_info=True)

    log.info("Detector ready: %s", app.state.detector.get_info())
    yield

    if app.state.db_enabled:
        from app.database.engine import dispose_engine

        await dispose_engine()
    log.info("Shutting down")


SERVICE_DESCRIPTION = """
Детекция русскоязычных текстов, сгенерированных ИИ (FastAPI + TF-IDF / трансформер / каскад).

**Ограничения и дисклеймер.** Сервис возвращает вероятностную оценку, а не доказательство
авторства. Это не проверка фактов и не оценка качества или содержания текста. Точность
снижается на коротких фрагментах (< `min_chars`), пост-отредактированных AI-текстах и
текстах от моделей, не представленных в обучающей выборке. Решения с правовыми или
дисциплинарными последствиями должны приниматься человеком с учётом дополнительного
контекста.

**Probes.** `GET /health` — liveness (всегда 200 пока процесс жив). `GET /ready` — readiness
(503 если детектор не загружен или БД недоступна при `DB_ENABLED=true`); пригоден для k8s
`readinessProbe`.
"""


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="AI Text Trust Service",
        description=SERVICE_DESCRIPTION,
        # Версия берётся из Settings: иначе Swagger UI и /openapi.json
        # покажут одну версию, а тег образа в проде — другую.
        version=settings.service_version,
        lifespan=lifespan,
    )

    # slowapi: обработчик 429 и app.state.limiter для декораторов в роутерах.
    from slowapi import _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS_ORIGINS — список через запятую из ENV. Пустая строка означает same-origin.
    # Пример для PWA на Vercel: "https://<app>.vercel.app,https://aitrust.example.ru".
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-Correlation-ID", "X-Request-ID"],
            expose_headers=["X-Correlation-ID"],
            max_age=600,
        )

    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(me_router)
    app.include_router(batch_me_router)  # /me/batches
    app.include_router(analyze_router)
    app.include_router(jobs_router)
    app.include_router(batch_router)  # /api/v1/batch/*
    app.include_router(extract_router)  # /api/v1/extract/text
    app.include_router(feedback_router)
    app.add_route("/metrics", metrics_endpoint, methods=["GET"])
    app.add_middleware(RequestLoggingMiddleware)
    return app


# Direct import for `uvicorn app.main:app`
app = create_app()
