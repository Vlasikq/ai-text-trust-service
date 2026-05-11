"""
Liveness и readiness endpoints.

GET /health — liveness: процесс жив, минимум полей. Бьётся локальным healthcheck'ом
              docker-compose. Всегда 200 если процесс отвечает.
GET /ready  — readiness: детектор загружен и БД (если включена) пингуется. Возвращает
              503 если что-то критичное не готово. Используется как readinessProbe в k8s.
"""

import logging

from fastapi import APIRouter, Request, Response
from sqlalchemy import text

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health(request: Request):
    """Liveness probe — минимум полей; всегда 200 если процесс отвечает."""
    settings = request.app.state.settings
    return {
        "status": "ok",
        "service_version": settings.service_version,
    }


async def _ping_db() -> bool:
    """SELECT 1 через engine; True если БД отвечает."""
    try:
        from app.database.engine import get_engine

        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        log.warning("DB readiness ping failed", exc_info=True)
        return False


@router.get("/ready")
async def ready(request: Request, response: Response):
    """Readiness probe — детектор + БД (если включена). Возвращает компонентные статусы.

    503 если: detector не готов ИЛИ db_enabled и БД не отвечает.
    """
    settings = request.app.state.settings
    detector = getattr(request.app.state, "detector", None)
    calibrator = getattr(request.app.state, "calibrator", None)
    explainer = getattr(request.app.state, "explainer", None)
    cache = getattr(request.app.state, "cache", None)
    db_enabled = getattr(request.app.state, "db_enabled", False)

    detector_ready = detector is not None and detector.is_ready()
    detector_info = detector.get_info() if detector is not None else None

    if db_enabled:
        db_ready = await _ping_db()
        db_status: dict = {"enabled": True, "ready": db_ready}
    else:
        db_ready = None  # not gating — disabled by config / unavailable at startup
        db_status = {"enabled": False, "ready": None}

    components = {
        "detector": {"ready": detector_ready, "info": detector_info},
        "db": db_status,
        "calibrator": {"ready": calibrator is not None},
        "explainer": {"ready": explainer is not None},
        "cache": cache.stats() if cache is not None else None,
    }

    is_ready = detector_ready and (not db_enabled or db_ready)
    if not is_ready:
        response.status_code = 503

    return {
        "status": "ready" if is_ready else "not_ready",
        "service_version": settings.service_version,
        "model_version": settings.model_version,
        "components": components,
    }
