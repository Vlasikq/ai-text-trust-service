"""Агрегация статистики `/me/stats` пятью GROUP BY-запросами по индексу
`(user_id, requested_at)`. Размер ответа БД ограничен числом групп, не числом
анализов — RAM растёт по количеству значений verdict / risk_level / detector.
Daily-bucket zero-padding делается в роутере, чтобы сервис не зависел от HTTP-схем.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import cast, func, select
from sqlalchemy.types import Date

from app.database.models import Analysis
from app.database.uow import UnitOfWork
from app.schemas import StatsPeriod

# Потолок для `period=all`: без верхней границы range scan для активного юзера
# деградирует без предела. Год покрывает реалистичный UI-кейс.
_HARD_LIMIT_DAYS_FOR_ALL = 365

_PERIOD_DAYS: dict[StatsPeriod, int] = {
    StatsPeriod.p7d: 7,
    StatsPeriod.p30d: 30,
    StatsPeriod.p90d: 90,
    StatsPeriod.p_all: _HARD_LIMIT_DAYS_FOR_ALL,
}


@dataclass
class UserStatsAggregates:
    """Сырьё для StatsResponse: агрегаты + только те даты, которые есть в БД."""

    total_scans: int
    by_verdict: dict[str, int]
    by_risk_level: dict[str, int]
    by_detector: dict[str, int]
    avg_confidence: float | None
    earliest: datetime | None
    daily: dict[date, dict[str, int]] = field(default_factory=dict)


async def compute_user_stats(
    uow: UnitOfWork,
    *,
    user_id: UUID,
    period: StatsPeriod,
    now: datetime | None = None,
) -> tuple[UserStatsAggregates, datetime, datetime]:
    """Считает агрегаты по analyses пользователя в окне `period`.

    Возвращает (агрегаты, period_start, period_end). Окно: `[now - period_days, now]`.
    Для `period=all` потолок 365 дней — см. `_HARD_LIMIT_DAYS_FOR_ALL`.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    period_end = now
    period_start = now - timedelta(days=_PERIOD_DAYS[period])

    base_filter = (
        Analysis.user_id == user_id,
        Analysis.requested_at >= period_start,
    )

    totals_stmt = select(
        func.count(Analysis.id),
        func.avg(Analysis.confidence),
        func.min(Analysis.requested_at),
    ).where(*base_filter)
    total, avg_conf, earliest = (await uow.session.execute(totals_stmt)).one()

    verdict_stmt = (
        select(Analysis.verdict, func.count(Analysis.id))
        .where(*base_filter, Analysis.verdict.isnot(None))
        .group_by(Analysis.verdict)
    )
    by_verdict = {v: c for v, c in (await uow.session.execute(verdict_stmt)).all()}

    risk_stmt = (
        select(Analysis.risk_level, func.count(Analysis.id))
        .where(*base_filter, Analysis.risk_level.isnot(None))
        .group_by(Analysis.risk_level)
    )
    by_risk = {r: c for r, c in (await uow.session.execute(risk_stmt)).all()}

    det_stmt = (
        select(Analysis.detector_used, func.count(Analysis.id))
        .where(*base_filter)
        .group_by(Analysis.detector_used)
    )
    by_detector = {d: c for d, c in (await uow.session.execute(det_stmt)).all()}

    # cast → Date в Postgres делает date_trunc в UTC; совпадает с тем,
    # как фронт показывает день по requested_at.
    day_col = cast(Analysis.requested_at, Date).label("day")
    daily_stmt = (
        select(day_col, Analysis.verdict, func.count(Analysis.id))
        .where(*base_filter)
        .group_by(day_col, Analysis.verdict)
    )
    daily: dict[date, dict[str, int]] = {}
    for day, verdict, count in (await uow.session.execute(daily_stmt)).all():
        bucket = daily.setdefault(day, {"total": 0, "ai": 0, "human": 0})
        bucket["total"] += count
        if verdict == "ai":
            bucket["ai"] += count
        elif verdict == "human":
            bucket["human"] += count

    agg = UserStatsAggregates(
        total_scans=int(total or 0),
        by_verdict=by_verdict,
        by_risk_level=by_risk,
        by_detector=by_detector,
        avg_confidence=float(avg_conf) if avg_conf is not None else None,
        earliest=earliest,
        daily=daily,
    )
    return agg, period_start, period_end
