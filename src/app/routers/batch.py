"""Batch-режим: загрузка CSV → пакетная обработка через очередь.

Endpoints:
  POST /api/v1/batch                       — создать batch из multipart CSV
  GET  /api/v1/batch/{id}                  — статус batch'а + skipped из парсинга
  GET  /api/v1/batch/{id}/results.csv      — отчёт CSV (StreamingResponse)
  GET  /api/v1/batch/{id}/results.json     — отчёт JSON

Privacy-by-design:
  - текст лежит в `analysis_jobs.input_text` только в inflight-фазе;
    воркер зануляет его в `mark_success` / `mark_error`.
  - отчёт содержит только результаты + `external_id`/`name` из исходного CSV;
    самих текстов в отчёте нет (см. явный комментарий в `_iter_csv_rows`).

Атомарность «1 active batch per user» — через partial unique index
`uq_one_active_batch_per_user` (см. миграцию 004). На втором POST мы ловим
IntegrityError → 409.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from typing import AsyncIterator
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from uuid_utils import uuid7

from app.auth.dependencies import get_current_user
from app.cache import text_hash
from app.config import DetectorType, get_settings
from app.database.models import (
    Analysis,
    AnalysisJob,
    BatchJob,
    BatchStatus,
    JobStatus,
    User,
)
from app.database.uow import UnitOfWork
from app.middleware.ratelimit import limiter
from app.schemas import (
    BatchCreatedResponse,
    BatchSkippedRow,
    BatchStatusResponse,
    BatchStatusValue,
)
from app.services.csv_parser import CsvParseError, parse_csv
from app.services.csv_writer import escape_row

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/batch", tags=["batch"])

# Лимиты: совпадают с дефолтами csv_parser, но передаются явно — иначе при
# изменении одного места поведение разъедется.
MAX_BATCH_SIZE_BYTES = 2 * 1024 * 1024
MAX_BATCH_ROWS = 100
MIN_TEXT_LEN = 300
MAX_TEXT_LEN = 8000


# Вспомогательные функции.


def _skipped_to_dicts(skipped) -> list[dict]:
    """SkippedRow dataclass → JSONB-совместимый dict (для batch_jobs.skipped_summary)."""
    return [
        {"row_num": s.row_num, "reason": s.reason, "external_id": s.external_id}
        for s in skipped
    ]


def _skipped_from_db(rows: list[dict] | None) -> list[BatchSkippedRow]:
    return [BatchSkippedRow(**r) for r in (rows or [])]


def _batch_status_response(batch: BatchJob) -> BatchStatusResponse:
    """Сборка JSON-ответа с готовыми URL'ами скачивания (если COMPLETED)."""
    completed = batch.status == BatchStatus.COMPLETED
    return BatchStatusResponse(
        batch_id=str(batch.id),
        file_name=batch.file_name,
        status=BatchStatusValue(batch.status),
        n_total=batch.n_total,
        n_completed=batch.n_completed,
        n_failed=batch.n_failed,
        detector_type=batch.detector_type,
        explain=batch.explain,
        created_at=batch.created_at.isoformat() if batch.created_at else "",
        completed_at=batch.completed_at.isoformat() if batch.completed_at else None,
        skipped=_skipped_from_db(batch.skipped_summary),
        results_csv_url=f"/api/v1/batch/{batch.id}/results.csv" if completed else None,
        results_json_url=f"/api/v1/batch/{batch.id}/results.json" if completed else None,
    )


# POST /api/v1/batch.


@router.post("", response_model=BatchCreatedResponse, status_code=202)
@limiter.limit(lambda: get_settings().rate_limit_batch)
async def create_batch(
    request: Request,
    file: UploadFile = File(...),
    detector_type: DetectorType = Form(default=DetectorType.tfidf),
    explain: bool = Form(default=False),
    current_user: User = Depends(get_current_user),
) -> BatchCreatedResponse:
    """Принять CSV → распарсить → bulk-вставить batch_jobs(1) + analysis_jobs(N).

    Лимиты: см. константы модуля. На втором активном batch'е у того же
    пользователя partial unique index выбрасывает IntegrityError → 409.
    """
    db_enabled = getattr(request.app.state, "db_enabled", False)
    if not db_enabled:
        raise HTTPException(status_code=503, detail="Database not available — batch disabled")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Файл пуст")

    try:
        parsed = parse_csv(
            raw,
            max_size_bytes=MAX_BATCH_SIZE_BYTES,
            max_rows=MAX_BATCH_ROWS,
            min_text_len=MIN_TEXT_LEN,
            max_text_len=MAX_TEXT_LEN,
        )
    except CsvParseError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not parsed.rows:
        raise HTTPException(
            status_code=400,
            detail="Ни одной валидной строки в CSV (все строки скипнуты или пусты)",
        )

    # Используем явные `pg_insert(...).values([...])`, а не ORM `add_all`:
    # IMV-sentinel matching SQLAlchemy ломается на UUID-PK при bulk insert
    # (uuid_utils.UUID ≠ uuid.UUID после round-trip через asyncpg). Прямой
    # INSERT VALUES обходит этот код и работает стабильно.
    batch_id_str = str(uuid7())
    skipped_dicts = _skipped_to_dicts(parsed.skipped)
    user_id_str = str(current_user.id)

    batch_values = {
        "id": batch_id_str,
        "user_id": user_id_str,
        "file_name": file.filename,
        "n_total": len(parsed.rows),
        "n_completed": 0,
        "n_failed": 0,
        "skipped_summary": skipped_dicts,
        "status": BatchStatus.PENDING,
        "detector_type": detector_type.value,
        "explain": explain,
    }
    job_values = [
        {
            "id": str(uuid7()),
            "request_id": str(uuid7()),
            "input_text": row.text,
            "text_hash": text_hash(row.text),
            "text_length": len(row.text),
            "detector_type": detector_type.value,
            "explain": explain,
            "status": JobStatus.PENDING,
            "user_id": user_id_str,
            "batch_id": batch_id_str,
            "batch_external_id": row.external_id,
            "batch_external_name": row.name,
        }
        for row in parsed.rows
    ]

    try:
        async with UnitOfWork() as uow:
            await uow.session.execute(pg_insert(BatchJob).values(batch_values))
            await uow.session.execute(pg_insert(AnalysisJob).values(job_values))
            await uow.commit()
    except IntegrityError as e:
        # Partial unique index `uq_one_active_batch_per_user` — единственный
        # конфликт, который ожидаем здесь. UNIQUE на request_id у jobs тоже
        # возможен, но крайне маловероятен (uuid7 коллизий).
        if "uq_one_active_batch_per_user" in str(e.orig):
            raise HTTPException(
                status_code=409,
                detail=(
                    "У вас уже есть активный batch. "
                    "Дождитесь завершения или скачайте результаты."
                ),
            ) from e
        raise

    log.info(
        "batch_created",
        extra={
            "batch_id": batch_id_str,
            "user_id": str(current_user.id),
            "n_total": len(parsed.rows),
            "n_skipped": len(parsed.skipped),
            "detector_type": detector_type.value,
        },
    )

    return BatchCreatedResponse(
        batch_id=batch_id_str,
        n_accepted=len(parsed.rows),
        n_skipped=len(parsed.skipped),
        skipped=_skipped_from_db(skipped_dicts),
        poll_url=f"/api/v1/batch/{batch_id_str}",
    )


# GET /api/v1/batch/{id}.


async def _load_owned_batch(request: Request, batch_id: str, user: User) -> BatchJob:
    """Загрузить batch с проверкой владельца. 404 для чужого/несуществующего."""
    db_enabled = getattr(request.app.state, "db_enabled", False)
    if not db_enabled:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        bid = UUID(batch_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Batch not found")

    async with UnitOfWork() as uow:
        b = (
            await uow.session.execute(select(BatchJob).where(BatchJob.id == bid))
        ).scalar_one_or_none()

    if b is None or str(b.user_id) != str(user.id):
        raise HTTPException(status_code=404, detail="Batch not found")
    return b


@router.get("/{batch_id}", response_model=BatchStatusResponse)
@limiter.limit(lambda: get_settings().rate_limit_me)
async def get_batch_status(
    request: Request,
    batch_id: str,
    current_user: User = Depends(get_current_user),
) -> BatchStatusResponse:
    batch = await _load_owned_batch(request, batch_id, current_user)
    return _batch_status_response(batch)


# Скачивание отчёта по результатам.


_REPORT_COLUMNS = (
    "row_num",
    "external_id",
    "external_name",
    "status",
    "verdict",
    "confidence",
    "risk_level",
    "detector_used",
    "inference_ms",
    "error",
)


async def _iter_report_rows(batch_id: UUID, user_id: str) -> AsyncIterator[dict]:
    """Стримит сводные строки отчёта.

    Privacy: исходный текст НЕ возвращается. В отчёте только результаты +
    `external_id`/`external_name` из CSV, чтобы клиент мог сматчить.

    Каждый child analysis_job → одна строка. Если SUCCESS — подгружаем
    `Analysis` через analysis_id. Если ERROR/PENDING/PROCESSING — заполняем
    статус-полем без verdict/confidence.
    """
    async with UnitOfWork() as uow:
        jobs = (
            await uow.session.execute(
                select(AnalysisJob)
                .where(
                    AnalysisJob.batch_id == batch_id,
                    AnalysisJob.user_id == UUID(user_id),
                )
                .order_by(AnalysisJob.created_at, AnalysisJob.id)
            )
        ).scalars().all()

        # Подтягиваем все Analysis оптом по id, чтобы не делать N+1.
        analysis_ids = [j.analysis_id for j in jobs if j.analysis_id is not None]
        analysis_by_id: dict = {}
        if analysis_ids:
            for a in (
                await uow.session.execute(
                    select(Analysis).where(Analysis.id.in_(analysis_ids))
                )
            ).scalars().all():
                analysis_by_id[str(a.id)] = a

    for idx, job in enumerate(jobs, start=1):
        record = {
            "row_num": idx,
            "external_id": job.batch_external_id,
            "external_name": job.batch_external_name,
            "status": job.status,
            "verdict": None,
            "confidence": None,
            "risk_level": None,
            "detector_used": None,
            "inference_ms": None,
            "error": None,
        }
        if job.status == JobStatus.SUCCESS and job.analysis_id is not None:
            a = analysis_by_id.get(str(job.analysis_id))
            if a is not None:
                record.update(
                    verdict=a.verdict,
                    confidence=a.confidence,
                    risk_level=a.risk_level,
                    detector_used=a.detector_used,
                    inference_ms=a.inference_ms,
                )
        elif job.status == JobStatus.ERROR:
            record["error"] = job.error_message
        yield record


def _csv_stream(rows: list[dict]) -> AsyncIterator[bytes]:
    """Сборка CSV без загрузки всех bytes сразу: yield-ом по строке."""

    async def gen():
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=_REPORT_COLUMNS)
        writer.writeheader()
        yield buf.getvalue().encode("utf-8")
        buf.seek(0)
        buf.truncate(0)

        for r in rows:
            writer.writerow(escape_row(r))
            data = buf.getvalue().encode("utf-8")
            buf.seek(0)
            buf.truncate(0)
            yield data

    return gen()


@router.get("/{batch_id}/results.csv")
@limiter.limit(lambda: get_settings().rate_limit_me)
async def get_results_csv(
    request: Request,
    batch_id: str,
    current_user: User = Depends(get_current_user),
):
    batch = await _load_owned_batch(request, batch_id, current_user)
    if batch.status != BatchStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=f"Batch ещё не завершён (status={batch.status})",
        )

    # Собираем rows один раз (n ≤ 100) и стримим. Для больших n стоит итератор,
    # но текущий лимит делает это избыточным.
    rows = [r async for r in _iter_report_rows(batch.id, str(current_user.id))]
    filename = f"batch_{batch.id}.csv"
    return StreamingResponse(
        _csv_stream(rows),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{batch_id}/results.json")
@limiter.limit(lambda: get_settings().rate_limit_me)
async def get_results_json(
    request: Request,
    batch_id: str,
    current_user: User = Depends(get_current_user),
):
    batch = await _load_owned_batch(request, batch_id, current_user)
    if batch.status != BatchStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=f"Batch ещё не завершён (status={batch.status})",
        )

    rows = [r async for r in _iter_report_rows(batch.id, str(current_user.id))]
    payload = {
        "batch_id": str(batch.id),
        "file_name": batch.file_name,
        "n_total": batch.n_total,
        "n_completed": batch.n_completed,
        "n_failed": batch.n_failed,
        "skipped": batch.skipped_summary,
        "results": rows,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    filename = f"batch_{batch.id}.json"
    return StreamingResponse(
        iter([body]),
        media_type="application/json; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# GET /me/batches.


# /me/batches лежит вне batch-роутера (префикс `/me`), а не `/api/v1/batch`.
me_router = APIRouter(prefix="/me", tags=["me"])


_CURSOR_SEP = "__"


def _parse_batch_cursor(cursor: str) -> tuple[str, UUID]:
    parts = cursor.split(_CURSOR_SEP, 1)
    if len(parts) != 2:
        raise HTTPException(status_code=400, detail="Невалидный курсор")
    try:
        return parts[0], UUID(parts[1])
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Невалидный курсор") from e


@me_router.get("/batches")
@limiter.limit(lambda: get_settings().rate_limit_me)
async def my_batches(
    request: Request,
    current_user: User = Depends(get_current_user),
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None),
):
    """История batch'ей пользователя. Cursor pagination по (created_at, id) DESC."""
    from app.schemas import BatchListItem, BatchListPage

    user_uuid = UUID(str(current_user.id))
    async with UnitOfWork() as uow:
        stmt = (
            select(BatchJob)
            .where(BatchJob.user_id == user_uuid)
            .order_by(BatchJob.created_at.desc(), BatchJob.id.desc())
            .limit(limit + 1)
        )
        if cursor:
            ts_str, cid = _parse_batch_cursor(cursor)
            from datetime import datetime as _dt

            try:
                cts = _dt.fromisoformat(ts_str)
            except ValueError:
                raise HTTPException(status_code=400, detail="Невалидный курсор")

            from sqlalchemy import and_, or_

            stmt = stmt.where(
                or_(
                    BatchJob.created_at < cts,
                    and_(BatchJob.created_at == cts, BatchJob.id < cid),
                )
            )
        rows = (await uow.session.execute(stmt)).scalars().all()

    items = [
        BatchListItem(
            batch_id=str(b.id),
            file_name=b.file_name,
            status=BatchStatusValue(b.status),
            n_total=b.n_total,
            n_completed=b.n_completed,
            n_failed=b.n_failed,
            created_at=b.created_at.isoformat() if b.created_at else "",
            completed_at=b.completed_at.isoformat() if b.completed_at else None,
        )
        for b in rows[:limit]
    ]
    next_cursor = (
        f"{rows[limit - 1].created_at.isoformat()}{_CURSOR_SEP}{rows[limit - 1].id}"
        if len(rows) > limit and items
        else None
    )
    return BatchListPage(items=items, next_cursor=next_cursor)


__all__ = ["router", "me_router"]
