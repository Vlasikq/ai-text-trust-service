"""
Background persistence — write analysis results to DB without blocking response.

On DB failure: write JSON to fallback log (zero data loss).
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from uuid_utils import uuid7

from app.cache import text_hash
from app.database.models import Analysis
from app.database.uow import UnitOfWork
from app.schemas import AnalyzeResponse

log = logging.getLogger(__name__)


def response_to_model(
    response: AnalyzeResponse,
    cleaned_text: str,
    detector_method: str,
    cascade_path: str | None,
    inference_ms: float,
    cached: bool,
    model_version: str,
    service_version: str,
    deployment_id: str | None = None,
    user_id: str | None = None,
) -> Analysis:
    """Convert API response + context into an ORM model."""
    explanation_dict = None
    if response.explanation:
        explanation_dict = response.explanation.model_dump()

    return Analysis(
        id=uuid7(),
        request_id=response.request_id,
        text_hash=text_hash(cleaned_text),
        text_length=response.text_length,
        status=response.status.value,
        verdict=response.verdict.value if response.verdict else None,
        confidence=response.confidence,
        risk_level=response.risk_level.value if response.risk_level else None,
        detector_used=detector_method,
        cascade_path=cascade_path,
        inference_ms=inference_ms,
        cached=cached,
        model_version=model_version,
        service_version=service_version,
        deployment_id=deployment_id,
        user_id=user_id,
        warnings=[w.value for w in response.warnings],
        explanation=explanation_dict,
        requested_at=datetime.now(timezone.utc),
    )


async def persist_with_fallback(
    analysis: Analysis,
    fallback_path: Path,
) -> None:
    """Try DB write; on failure, append JSON to fallback log."""
    try:
        async with UnitOfWork() as uow:
            await uow.analyses.save(analysis)
            await uow.commit()
    except Exception as e:
        log.error(
            "db_write_failed request_id=%s error=%s",
            analysis.request_id,
            str(e),
        )
        _write_fallback(analysis, fallback_path)


def _write_fallback(analysis: Analysis, path: Path) -> None:
    """Append analysis as JSON line to fallback log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "request_id": str(analysis.request_id),
        "text_hash": analysis.text_hash,
        "text_length": analysis.text_length,
        "status": analysis.status,
        "verdict": analysis.verdict,
        "confidence": analysis.confidence,
        "risk_level": analysis.risk_level,
        "detector_used": analysis.detector_used,
        "cascade_path": analysis.cascade_path,
        "inference_ms": analysis.inference_ms,
        "cached": analysis.cached,
        "model_version": analysis.model_version,
        "service_version": analysis.service_version,
        "warnings": analysis.warnings,
        "requested_at": analysis.requested_at.isoformat() if analysis.requested_at else None,
        "fallback_reason": "db_write_failed",
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log.info("Fallback log written: %s", path)
    except Exception:
        log.critical("Fallback log write also failed!", exc_info=True)
