"""
POST /api/v1/feedback  — user correction endpoint.
GET  /api/v1/stats     — aggregate analytics (admin-only).
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.auth.dependencies import get_current_user
from app.config import get_settings
from app.database.models import Feedback, User, UserRole
from app.database.uow import UnitOfWork
from app.middleware.ratelimit import limiter

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


class FeedbackRequest(BaseModel):
    analysis_id: str | None = None
    text_hash: str | None = None
    user_verdict: str = Field(..., pattern="^(human|ai)$")
    source: str = "api"
    comment: str | None = None


class FeedbackResponse(BaseModel):
    id: str
    status: str = "saved"


@router.post("/feedback", response_model=FeedbackResponse)
@limiter.limit(lambda: get_settings().rate_limit_feedback)
async def submit_feedback(body: FeedbackRequest, request: Request):
    db_enabled = getattr(request.app.state, "db_enabled", False)
    if not db_enabled:
        raise HTTPException(status_code=503, detail="Database not available")

    async with UnitOfWork() as uow:
        fb = Feedback(
            analysis_id=body.analysis_id,
            text_hash=body.text_hash,
            user_verdict=body.user_verdict,
            source=body.source,
            comment=body.comment,
        )
        await uow.feedback.save(fb)
        await uow.commit()

    return FeedbackResponse(id=str(fb.id))


@router.get("/stats")
async def get_stats(request: Request, user: User = Depends(get_current_user)):
    """Агрегаты по анализам — только для admin-роли.

    Эндпоинт раскрывает счётчики верных/неверных детекций и распределения по
    cascade-пути; на публичном проде это утечка операционных метрик. См.
    SECURITY review «19»: было public, стало admin-only.
    """
    if user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin role required")

    db_enabled = getattr(request.app.state, "db_enabled", False)
    if not db_enabled:
        raise HTTPException(status_code=503, detail="Database not available")

    async with UnitOfWork() as uow:
        stats = await uow.analyses.get_stats()

    return stats
