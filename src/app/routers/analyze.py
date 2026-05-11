"""Синхронный endpoint детекции `POST /api/v1/analyze`.

Тонкий слой: разворачивает HTTP-контекст, вызывает `services.detection.run_detection`,
снимает метрики и фоновой задачей сохраняет результат в БД. Та же логика
используется асинхронным воркером из `worker.py`.

Liveness и readiness вынесены в `app.routers.health`.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, Request

from app.auth.dependencies import get_current_user_optional
from app.config import get_settings
from app.database.models import User
from app.logging_config import get_correlation_id
from app.middleware.metrics import record_request
from app.middleware.ratelimit import limiter
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    SegmentScoreItem,
    SegmentScoringRequest,
    SegmentScoringResponse,
    SegmentSummary,
    Status,
)
from app.services.detection import DetectionContext, run_detection
from app.services.segment_scoring import detect_transition, score_segments

log = logging.getLogger(__name__)

router = APIRouter()


def _build_context(request: Request) -> DetectionContext:
    """Собирает DI-контейнер из `app.state`: lifespan уже всё подгрузил."""
    state = request.app.state
    return DetectionContext(
        settings=state.settings,
        preprocessor=state.preprocessor,
        detector=state.detector,
        calibrator=getattr(state, "calibrator", None),
        explainer=getattr(state, "explainer", None),
        cache=getattr(state, "cache", None),
    )


# Лимит общий на endpoint. Разделение анонимного и авторизованного трафика
# уже сделано в `key_func`: аноним попадает в бакет "ip:<x>", авторизованный —
# в "auth:<token-prefix>", счётчики независимы. Провайдер лимита в slowapi
# вызывается без аргументов и не видит request, поэтому выбрать разный лимит
# по типу клиента здесь нельзя; ставим общий жёсткий потолок.
@router.post("/api/v1/analyze", response_model=AnalyzeResponse)
@limiter.limit(lambda: get_settings().rate_limit_analyze)
async def analyze(
    body: AnalyzeRequest,
    request: Request,
    bg: BackgroundTasks,
    current_user: User | None = Depends(get_current_user_optional),
):
    # request_id совпадает с correlation_id (заголовок X-Correlation-ID или новый UUID4).
    # Если middleware не отработал (тесты с голым TestClient) — генерируем сами.
    request_id = get_correlation_id() or str(uuid.uuid4())
    ctx = _build_context(request)

    timeout_s = ctx.settings.cascade_timeout_ms / 1000

    outcome = await run_detection(
        ctx,
        body.text,
        request_id=request_id,
        explain=body.explain,
        debug=body.debug,
        timeout_s=timeout_s,
    )

    response = outcome.response

    # Метрики снимаются только для успешных проходов;
    # статусы NO_DECISION и ERROR остаются в access-логе.
    if response.status == Status.success:
        record_request(
            status=response.status.value,
            risk_level=response.risk_level.value if response.risk_level else None,
            confidence=response.confidence or 0.0,
            latency_s=response.processing_time_ms / 1000,
            cascade_path=outcome.detection.metadata.get("cascade_path") if outcome.detection else None,
        )

    # Запись в БД идёт фоновой задачей, если БД включена и есть результат детекции.
    db_enabled = getattr(request.app.state, "db_enabled", False)
    if db_enabled and outcome.detection is not None:
        from app.database.persist import persist_with_fallback, response_to_model

        deployment_id = getattr(request.app.state, "deployment_id", None)
        analysis_model = response_to_model(
            response=response,
            cleaned_text=outcome.preprocessing.text,
            detector_method=outcome.detection.method,
            cascade_path=outcome.detection.metadata.get("cascade_path"),
            inference_ms=outcome.detection.inference_ms,
            cached=outcome.cached_hit,
            model_version=ctx.settings.model_version,
            service_version=ctx.settings.service_version,
            deployment_id=deployment_id,
            user_id=str(current_user.id) if current_user else None,
        )
        bg.add_task(persist_with_fallback, analysis_model, ctx.settings.db_fallback_log_path)

    return response


# Посегментная оценка по предложениям (POC).


@router.post(
    "/api/v1/analyze/segments",
    response_model=SegmentScoringResponse,
    summary="Покадровая оценка текста (экспериментально)",
)
@limiter.limit(lambda: get_settings().rate_limit_analyze)
async def analyze_segments(
    body: SegmentScoringRequest,
    request: Request,
) -> SegmentScoringResponse:
    """Скользящие окна по предложениям → P(AI) для каждого окна.

    Не сохраняется в БД — дешёвая визуализация для одиночного запроса. Детектор
    берётся из `app.state.detector` (тот же, что в /analyze). Калибратор и
    explanation НЕ применяются — segments-API оптимизирован под скорость и
    обзорность, а не под верификацию.
    """
    detector = request.app.state.detector

    def predict_fn(window_text: str) -> float:
        return float(detector.predict(window_text).prob_ai)

    scores = score_segments(
        body.text,
        predict_fn,
        window_size=body.window_size,
        step=body.step,
    )

    if scores:
        probs = [s.prob_ai for s in scores]
        summary = SegmentSummary(
            n_segments=len(scores),
            mean_prob=sum(probs) / len(probs),
            min_prob=min(probs),
            max_prob=max(probs),
            n_high=sum(1 for p in probs if p >= 0.7),
            n_medium=sum(1 for p in probs if 0.3 <= p < 0.7),
            n_low=sum(1 for p in probs if p < 0.3),
            transition_detected=detect_transition(probs),
        )
    else:
        summary = SegmentSummary(n_segments=0)

    return SegmentScoringResponse(
        text_length=len(body.text),
        window_size=body.window_size,
        step=body.step,
        segments=[
            SegmentScoreItem(
                start_sent=s.start_sent,
                end_sent=s.end_sent,
                start_char=s.start_char,
                end_char=s.end_char,
                prob_ai=s.prob_ai,
            )
            for s in scores
        ],
        summary=summary,
    )
