"""POST /api/v1/extract/text — извлечь plain-text из загруженного TXT/DOCX/PDF.

Используется в PWA при drag-and-drop файла на форму анализа: вместо того чтобы
просить пользователя копировать текст из Word, мы парсим его на бэкенде.

Privacy: содержимое не сохраняется в БД и не логируется. Возвращается клиенту,
который сам отправит его в `/analyze`.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.config import get_settings
from app.middleware.ratelimit import limiter
from app.services.file_extractor import ExtractError, extract

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/extract", tags=["extract"])

MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 МБ


class ExtractTextResponse(BaseModel):
    text: str
    source_format: str
    char_count: int
    warnings: list[str] = []


@router.post("/text", response_model=ExtractTextResponse)
@limiter.limit(lambda: get_settings().rate_limit_extract)
async def extract_text(
    request: Request,
    file: UploadFile = File(...),
) -> ExtractTextResponse:
    """Загрузить TXT/DOCX/PDF → вернуть plain-text. 5 МБ лимит на файл."""
    raw = await file.read()
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Файл превышает лимит {MAX_FILE_BYTES // 1024 // 1024} МБ",
        )

    try:
        result = extract(
            raw,
            content_type=file.content_type,
            filename=file.filename,
        )
    except ExtractError as e:
        raise HTTPException(status_code=400, detail=str(e))

    log.info(
        "file_extracted",
        extra={
            "source_format": result.source_format,
            "char_count": result.char_count,
            "warnings": len(result.warnings),
        },
    )
    return ExtractTextResponse(
        text=result.text,
        source_format=result.source_format,
        char_count=result.char_count,
        warnings=result.warnings,
    )
