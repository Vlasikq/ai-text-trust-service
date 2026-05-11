"""ЛК (личный кабинет): GET /me, GET /me/analyses, GET /me/analyses/{id}, GET /me/stats.

Защищены `get_current_user` (401 если bearer-токен невалиден / expired).
Курсорная пагинация по `(requested_at DESC, id DESC)` — id-tiebreaker
гарантирует стабильный порядок при коллизиях timestamp (concurrent inserts
в один и тот же миллисекунд / batch-импорт). См. SECURITY review «3».
"""

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import and_, or_, select

from app.auth.dependencies import get_current_user
from app.config import get_settings
from app.database.models import Analysis, User
from app.database.uow import UnitOfWork
from app.middleware.ratelimit import limiter
from app.schemas import (
    AnalysisDetailResponse,
    AnalysisHistoryItem,
    AnalysisHistoryPage,
    DailyBucket,
    Explanation,
    RiskLevel,
    StatsPeriod,
    StatsResponse,
    StyleMarker,
    UserPublic,
    Verdict,
    WarningCode,
)
from app.schemas import (
    Status as AnalysisStatus,
)

router = APIRouter(prefix="/me", tags=["me"])

# Разделитель частей курсора. Двойное `__` исключает коллизию с ISO-форматом
# (он использует одиночные `:`/`-`/`T`) и uuid (`-`). Не URL-encoded — его
# никогда не разбирают как часть пути.
_CURSOR_SEP = "__"


def _user_to_public(user: User) -> UserPublic:
    return UserPublic(
        id=str(user.id),
        email=user.email,
        role=user.role,
        status=user.status,
        created_at=user.created_at.isoformat() if user.created_at else "",
        last_login_at=user.last_login_at.isoformat() if user.last_login_at else None,
    )


def _parse_cursor(cursor: str) -> tuple[datetime, UUID]:
    """Разобрать `{iso_ts}__{uuid}` → (timestamp, id). 400 при битом формате."""
    parts = cursor.split(_CURSOR_SEP, 1)
    if len(parts) != 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"cursor must be '<iso_ts>{_CURSOR_SEP}<uuid>'",
        )
    try:
        ts = datetime.fromisoformat(parts[0])
        rid = UUID(parts[1])
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid cursor: expected ISO-8601 timestamp + UUID",
        ) from e
    return ts, rid


def _build_cursor(row: Analysis) -> str:
    return f"{row.requested_at.isoformat()}{_CURSOR_SEP}{row.id}"


@router.get("", response_model=UserPublic, summary="Текущий пользователь")
@limiter.limit(lambda: get_settings().rate_limit_me)
async def me(request: Request, user: User = Depends(get_current_user)) -> UserPublic:
    return _user_to_public(user)


@router.get(
    "/analyses",
    response_model=AnalysisHistoryPage,
    summary="История анализов пользователя (cursor pagination)",
)
@limiter.limit(lambda: get_settings().rate_limit_me)
async def my_analyses(
    request: Request,
    user: User = Depends(get_current_user),
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(
        default=None,
        description="Курсор предыдущей страницы в формате '<iso_ts>__<uuid>' "
        "(значение `next_cursor` из предыдущего ответа).",
    ),
) -> AnalysisHistoryPage:
    """Возвращает анализы пользователя в порядке (requested_at, id) DESC.

    Композитный курсор закрывает дефект: при коллизии `requested_at` (batch /
    миллисекундное разрешение / тестовая фиксация времени) старый одиночный
    курсор пропускал или дублировал записи. Теперь tiebreaker по uuid7 (он
    хронологический), порядок детерминирован.
    """
    cursor_ts: datetime | None = None
    cursor_id: UUID | None = None
    if cursor:
        cursor_ts, cursor_id = _parse_cursor(cursor)

    async with UnitOfWork() as uow:
        stmt = (
            select(Analysis)
            .where(Analysis.user_id == UUID(str(user.id)))
            .order_by(Analysis.requested_at.desc(), Analysis.id.desc())
            .limit(limit + 1)  # +1 чтобы понять, есть ли next page
        )
        if cursor_ts is not None and cursor_id is not None:
            # Композитное условие "(requested_at, id) < (cursor_ts, cursor_id)"
            # в DESC-порядке: либо строго раньше по timestamp, либо тот же
            # timestamp и меньший id.
            stmt = stmt.where(
                or_(
                    Analysis.requested_at < cursor_ts,
                    and_(
                        Analysis.requested_at == cursor_ts,
                        Analysis.id < cursor_id,
                    ),
                )
            )

        rows = (await uow.session.execute(stmt)).scalars().all()

    items = [
        AnalysisHistoryItem(
            id=str(a.id),
            request_id=str(a.request_id),
            status=AnalysisStatus(a.status),
            verdict=a.verdict,
            confidence=a.confidence,
            risk_level=a.risk_level,
            detector_used=a.detector_used,
            text_length=a.text_length,
            requested_at=a.requested_at.isoformat(),
        )
        for a in rows[:limit]
    ]
    next_cursor = _build_cursor(rows[limit - 1]) if len(rows) > limit and items else None
    return AnalysisHistoryPage(items=items, next_cursor=next_cursor)


@router.get(
    "/analyses/{analysis_id}",
    response_model=AnalysisDetailResponse,
    summary="Полная карточка анализа",
)
@limiter.limit(lambda: get_settings().rate_limit_me)
async def my_analysis_detail(
    request: Request,
    analysis_id: str,
    user: User = Depends(get_current_user),
) -> AnalysisDetailResponse:
    """Возвращает карточку анализа по id с проверкой владельца.

    Чужие и несуществующие записи отдаются одинаково как 404 — это сознательно:
    403 утекал бы факт существования записи у другого пользователя (enumeration).
    """
    try:
        aid = UUID(analysis_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")

    async with UnitOfWork() as uow:
        stmt = select(Analysis).where(Analysis.id == aid)
        analysis = (await uow.session.execute(stmt)).scalar_one_or_none()

    if analysis is None or str(analysis.user_id) != str(user.id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")

    explanation: Explanation | None = None
    if analysis.explanation:
        markers = analysis.explanation.get("top_markers", []) or []
        explanation = Explanation(
            top_markers=[StyleMarker(**m) for m in markers],
            summary=analysis.explanation.get("summary", ""),
        )

    return AnalysisDetailResponse(
        id=str(analysis.id),
        request_id=str(analysis.request_id),
        status=AnalysisStatus(analysis.status),
        verdict=Verdict(analysis.verdict) if analysis.verdict else None,
        confidence=analysis.confidence,
        risk_level=RiskLevel(analysis.risk_level) if analysis.risk_level else None,
        detector_used=analysis.detector_used,
        text_length=analysis.text_length,
        requested_at=analysis.requested_at.isoformat(),
        inference_ms=analysis.inference_ms,
        cached=analysis.cached,
        cascade_path=analysis.cascade_path,
        warnings=[WarningCode(w) for w in (analysis.warnings or [])],
        explanation=explanation,
        disclaimer=get_settings().disclaimer_text,
    )


_PERIOD_DAYS: dict[StatsPeriod, int | None] = {
    StatsPeriod.p7d: 7,
    StatsPeriod.p30d: 30,
    StatsPeriod.p90d: 90,
    StatsPeriod.p_all: None,
}


def _zero_pad_daily(
    rows_by_date: dict[date, dict[str, int]],
    start: date,
    end: date,
) -> list[DailyBucket]:
    """Заполнить пропущенные дни нулями. start/end включительно."""
    out: list[DailyBucket] = []
    cur = start
    while cur <= end:
        bucket = rows_by_date.get(cur, {"total": 0, "ai": 0, "human": 0})
        out.append(
            DailyBucket(
                date=cur.isoformat(),
                total=bucket["total"],
                ai=bucket["ai"],
                human=bucket["human"],
            )
        )
        cur += timedelta(days=1)
    return out


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Персональная статистика анализов",
)
@limiter.limit(lambda: get_settings().rate_limit_me)
async def my_stats(
    request: Request,
    user: User = Depends(get_current_user),
    period: StatsPeriod = Query(default=StatsPeriod.p30d),
) -> StatsResponse:
    """Агрегаты по анализам пользователя за окно `period`.

    Выгружаем все Analysis пользователя в окне одной выборкой и собираем
    агрегаты в Python — для типичной нагрузки ЛК (≤ нескольких сотен записей)
    это проще, чем 4 GROUP BY-запроса. Daily zero-padded, чтобы фронт не
    делал mock-данных под пустые дни.
    """
    now = datetime.now(timezone.utc)
    period_end = now
    days = _PERIOD_DAYS[period]
    period_start: datetime | None = (now - timedelta(days=days)) if days else None

    user_uuid = UUID(str(user.id))
    async with UnitOfWork() as uow:
        stmt = select(Analysis).where(Analysis.user_id == user_uuid)
        if period_start is not None:
            stmt = stmt.where(Analysis.requested_at >= period_start)
        rows = (await uow.session.execute(stmt)).scalars().all()

    by_verdict: dict[str, int] = defaultdict(int)
    by_risk: dict[str, int] = defaultdict(int)
    by_detector: dict[str, int] = defaultdict(int)
    confidences: list[float] = []
    daily_raw: dict[date, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "ai": 0, "human": 0}
    )
    earliest: datetime | None = None

    for a in rows:
        if a.verdict:
            by_verdict[a.verdict] += 1
        if a.risk_level:
            by_risk[a.risk_level] += 1
        by_detector[a.detector_used] += 1
        if a.confidence is not None:
            confidences.append(a.confidence)

        d = a.requested_at.astimezone(timezone.utc).date()
        bucket = daily_raw[d]
        bucket["total"] += 1
        if a.verdict == "ai":
            bucket["ai"] += 1
        elif a.verdict == "human":
            bucket["human"] += 1

        if earliest is None or a.requested_at < earliest:
            earliest = a.requested_at

    # period_start для "all" — самый ранний анализ или сегодня (если данных нет).
    if period_start is None:
        period_start = earliest if earliest is not None else period_end

    daily = _zero_pad_daily(daily_raw, period_start.date(), period_end.date())
    avg_conf = sum(confidences) / len(confidences) if confidences else None

    return StatsResponse(
        period=period,
        period_start=period_start.isoformat(),
        period_end=period_end.isoformat(),
        total_scans=len(rows),
        by_verdict=dict(by_verdict),
        by_risk_level=dict(by_risk),
        by_detector=dict(by_detector),
        avg_confidence=avg_conf,
        daily=daily,
    )
