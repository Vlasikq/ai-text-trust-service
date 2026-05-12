"""
POST /api/v1/jobs        — создать async-job для cascade-инференса.
GET  /api/v1/jobs/{id}   — статус + результат job-а.

Async-путь нужен потому что cascade в серой зоне на CPU занимает 5-15 секунд.
Sync `/api/v1/analyze` остаётся для уверенных-зон/TF-IDF only.

Worker (`src/app/worker.py`) опрашивает таблицу через `FOR UPDATE SKIP LOCKED`
и записывает результат в `analyses` + связывает с job через `analysis_id` FK.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from uuid_utils import uuid7

from app.auth.dependencies import get_current_user_optional
from app.cache import text_hash
from app.config import get_settings
from app.database.models import AnalysisJob, JobStatus, User
from app.database.uow import UnitOfWork
from app.logging_config import get_correlation_id
from app.middleware.ratelimit import limiter
from app.schemas import (
    AnalyzeResponse,
    JobCreatedResponse,
    JobCreateRequest,
    JobStatusResponse,
    JobStatusValue,
    RiskLevel,
    Status,
    Verdict,
    WarningCode,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


# Один лимит на endpoint — anon/auth разделены через key_func (см. analyze.py).
@router.post("/jobs", response_model=JobCreatedResponse, status_code=202)
@limiter.limit(lambda: get_settings().rate_limit_jobs)
async def create_job(
    body: JobCreateRequest,
    request: Request,
    current_user: User | None = Depends(get_current_user_optional),
) -> JobCreatedResponse:
    """Создать async-job. Возвращает 202 Accepted + poll_url для GET /jobs/{id}.

    Idempotent по `X-Correlation-ID` (== request_id): повторный POST с тем же id
    возвращает существующий job (через UNIQUE-constraint на request_id).
    """
    db_enabled = getattr(request.app.state, "db_enabled", False)
    if not db_enabled:
        raise HTTPException(status_code=503, detail="Database not available — async jobs disabled")

    request_id = get_correlation_id() or str(uuid.uuid4())
    settings = request.app.state.settings

    job = AnalysisJob(
        id=uuid7(),
        request_id=request_id,
        input_text=body.text,
        text_hash=text_hash(body.text),
        text_length=len(body.text),
        detector_type=settings.detector_type.value,
        explain=body.explain,
        status=JobStatus.PENDING,
        user_id=str(current_user.id) if current_user else None,
    )

    async with UnitOfWork() as uow:
        saved = await uow.jobs.create(job)
        await uow.commit()

    log.info("job_created", extra={"job_id": str(saved.id), "request_id": request_id})

    return JobCreatedResponse(
        job_id=str(saved.id),
        request_id=str(saved.request_id),
        status=JobStatusValue(saved.status),
        poll_url=f"/api/v1/jobs/{saved.id}",
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
@limiter.limit(lambda: get_settings().rate_limit_me)
async def get_job(
    job_id: str,
    request: Request,
    current_user: User | None = Depends(get_current_user_optional),
) -> JobStatusResponse:
    """Статус job-а. Если SUCCESS — подгружаем `Analysis` и возвращаем `AnalyzeResponse`.

    Анонимный job (user_id=None) виден любому по UUID: иначе анонимный клиент
    не сможет забрать свой результат. Авторский job — только владельцу; чужие
    и несуществующие отдаются как 404, чтобы не палить факт существования.
    """
    db_enabled = getattr(request.app.state, "db_enabled", False)
    if not db_enabled:
        raise HTTPException(status_code=503, detail="Database not available")

    async with UnitOfWork() as uow:
        job = await uow.jobs.get_by_id(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        if job.user_id is not None and (
            current_user is None or str(current_user.id) != str(job.user_id)
        ):
            raise HTTPException(status_code=404, detail="Job not found")

        result_payload: AnalyzeResponse | None = None
        if job.status == JobStatus.SUCCESS and job.analysis_id is not None:
            analysis = await uow.analyses.get_by_request_id(job.request_id)
            if analysis is not None:
                result_payload = _analysis_to_response(
                    analysis, settings=request.app.state.settings
                )

    return JobStatusResponse(
        job_id=str(job.id),
        request_id=str(job.request_id),
        status=JobStatusValue(job.status),
        created_at=job.created_at.isoformat() if job.created_at else "",
        started_at=job.started_at.isoformat() if job.started_at else None,
        finished_at=job.finished_at.isoformat() if job.finished_at else None,
        result=result_payload,
        error=job.error_message,
    )


def _analysis_to_response(analysis, settings) -> AnalyzeResponse:
    """Преобразовать Analysis ORM в AnalyzeResponse для polling-ответа.

    Method scores и Explanation восстанавливаются из JSONB (если были сохранены).
    """
    from app.schemas import Explanation, StyleMarker

    explanation = None
    if analysis.explanation:
        explanation = Explanation(
            top_markers=[StyleMarker(**m) for m in analysis.explanation.get("top_markers", [])],
            summary=analysis.explanation.get("summary", ""),
        )

    # method_scores вне JSONB не сохраняется; debug-режим в jobs не используется.
    method_scores = None

    return AnalyzeResponse(
        request_id=str(analysis.request_id),
        status=Status(analysis.status),
        verdict=Verdict(analysis.verdict) if analysis.verdict else None,
        confidence=analysis.confidence,
        risk_level=RiskLevel(analysis.risk_level) if analysis.risk_level else None,
        explanation=explanation,
        method_scores=method_scores,
        text_length=analysis.text_length,
        processing_time_ms=analysis.inference_ms,
        warnings=[WarningCode(w) for w in (analysis.warnings or [])],
        disclaimer=settings.disclaimer_text,
    )
