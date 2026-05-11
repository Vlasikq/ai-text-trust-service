"""
Repository pattern — data access layer for analyses, feedback, deployments, jobs.

Each repository takes an AsyncSession and provides typed CRUD + queries.
ON CONFLICT DO NOTHING for idempotent writes.
"""

from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import (
    Analysis,
    AnalysisJob,
    BatchJob,
    Feedback,
    JobStatus,
    ModelDeployment,
    RefreshToken,
    User,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AnalysisRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def save(self, analysis: Analysis) -> bool:
        """Insert analysis. Returns False if request_id already exists."""
        stmt = (
            pg_insert(Analysis)
            .values(
                id=analysis.id,
                request_id=analysis.request_id,
                text_hash=analysis.text_hash,
                text_length=analysis.text_length,
                status=analysis.status,
                verdict=analysis.verdict,
                confidence=analysis.confidence,
                risk_level=analysis.risk_level,
                detector_used=analysis.detector_used,
                cascade_path=analysis.cascade_path,
                inference_ms=analysis.inference_ms,
                cached=analysis.cached,
                model_version=analysis.model_version,
                service_version=analysis.service_version,
                deployment_id=analysis.deployment_id,
                user_id=analysis.user_id,
                warnings=analysis.warnings,
                explanation=analysis.explanation,
                requested_at=analysis.requested_at,
            )
            .on_conflict_do_nothing(index_elements=["request_id"])
        )
        result = await self._session.execute(stmt)
        return result.rowcount > 0

    async def get(self, analysis_id: UUID) -> Analysis | None:
        """Получить Analysis по PK."""
        return await self._session.get(Analysis, analysis_id)

    async def get_history_page(
        self,
        user_id: UUID,
        *,
        limit: int,
        cursor_ts: datetime | None = None,
        cursor_id: UUID | None = None,
    ) -> list[Analysis]:
        """Composite-cursor пагинация по (requested_at, id) DESC.

        Возвращает до `limit+1` строк, чтобы вызывающий понял, есть ли next page.
        Tiebreaker по id — стабильный порядок при коллизиях timestamp (batch insert).
        """
        from sqlalchemy import and_, or_

        stmt = (
            select(Analysis)
            .where(Analysis.user_id == user_id)
            .order_by(Analysis.requested_at.desc(), Analysis.id.desc())
            .limit(limit + 1)
        )
        if cursor_ts is not None and cursor_id is not None:
            stmt = stmt.where(
                or_(
                    Analysis.requested_at < cursor_ts,
                    and_(
                        Analysis.requested_at == cursor_ts,
                        Analysis.id < cursor_id,
                    ),
                )
            )
        return list((await self._session.execute(stmt)).scalars().all())

    async def get_by_request_id(self, request_id: UUID) -> Analysis | None:
        stmt = select(Analysis).where(Analysis.request_id == request_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_text_hash(self, text_hash: str) -> list[Analysis]:
        stmt = (
            select(Analysis)
            .where(Analysis.text_hash == text_hash)
            .order_by(Analysis.requested_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_stats(self, since: datetime | None = None) -> dict:
        """Aggregate stats: count by risk_level, detector, cascade_path."""
        base = select(Analysis)
        if since:
            base = base.where(Analysis.requested_at >= since)

        # Total count
        total_q = select(func.count(Analysis.id))
        if since:
            total_q = total_q.where(Analysis.requested_at >= since)
        total = (await self._session.execute(total_q)).scalar() or 0

        # By risk level
        risk_q = select(Analysis.risk_level, func.count()).group_by(Analysis.risk_level)
        if since:
            risk_q = risk_q.where(Analysis.requested_at >= since)
        risk_rows = (await self._session.execute(risk_q)).all()

        # By detector
        det_q = select(Analysis.detector_used, func.count()).group_by(Analysis.detector_used)
        if since:
            det_q = det_q.where(Analysis.requested_at >= since)
        det_rows = (await self._session.execute(det_q)).all()

        # By cascade path
        casc_q = select(Analysis.cascade_path, func.count()).group_by(Analysis.cascade_path)
        if since:
            casc_q = casc_q.where(Analysis.requested_at >= since)
        casc_rows = (await self._session.execute(casc_q)).all()

        return {
            "total": total,
            "by_risk_level": {r: c for r, c in risk_rows},
            "by_detector": {d: c for d, c in det_rows},
            "by_cascade_path": {p: c for p, c in casc_rows},
        }


class FeedbackRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def save(self, feedback: Feedback) -> None:
        self._session.add(feedback)

    async def get_by_analysis_id(self, analysis_id: UUID) -> list[Feedback]:
        stmt = select(Feedback).where(Feedback.analysis_id == analysis_id)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_text_hash(self, text_hash: str) -> list[Feedback]:
        stmt = select(Feedback).where(Feedback.text_hash == text_hash)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())


class JobRepository:
    """Async-job queue поверх PostgreSQL: SELECT ... FOR UPDATE SKIP LOCKED.

    Idempotent insert по UNIQUE(request_id) — повторный POST /jobs возвращает существующий job.
    Privacy: `input_text` обнуляется в `mark_success` / `mark_error`.
    """

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, job: AnalysisJob) -> AnalysisJob:
        """Insert new job. На UNIQUE-конфликте по request_id возвращает существующий."""
        stmt = (
            pg_insert(AnalysisJob)
            .values(
                id=job.id,
                request_id=job.request_id,
                input_text=job.input_text,
                text_hash=job.text_hash,
                text_length=job.text_length,
                detector_type=job.detector_type,
                explain=job.explain,
                status=job.status,
                user_id=job.user_id,
            )
            .on_conflict_do_nothing(index_elements=["request_id"])
        )
        await self._session.execute(stmt)
        # Возвращаем актуальную запись (либо вставленную, либо уже существующую)
        existing = await self.get_by_request_id(job.request_id)
        assert existing is not None, "after insert/conflict the row must exist"
        return existing

    async def get_by_id(self, job_id: UUID) -> AnalysisJob | None:
        stmt = select(AnalysisJob).where(AnalysisJob.id == job_id)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def get_by_request_id(self, request_id: UUID | str) -> AnalysisJob | None:
        stmt = select(AnalysisJob).where(AnalysisJob.request_id == request_id)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def claim_next(self, worker_id: str) -> AnalysisJob | None:
        """Worker берёт следующий PENDING-job через FOR UPDATE SKIP LOCKED.

        Атомарно: SELECT с блокировкой → UPDATE статуса в PROCESSING. Возврат
        внутри одной транзакции — другой worker этот job уже не возьмёт.
        """
        stmt = (
            select(AnalysisJob)
            .where(AnalysisJob.status == JobStatus.PENDING)
            .order_by(AnalysisJob.created_at)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        job = (await self._session.execute(stmt)).scalar_one_or_none()
        if job is None:
            return None
        job.status = JobStatus.PROCESSING
        job.worker_id = worker_id
        job.started_at = _utcnow()
        await self._session.flush()
        return job

    async def mark_success(self, job: AnalysisJob, analysis_id: UUID | str) -> None:
        """Завершить job успешно: занулить input_text (privacy), привязать analysis_id."""
        job.status = JobStatus.SUCCESS
        job.analysis_id = analysis_id
        job.finished_at = _utcnow()
        job.input_text = None  # privacy: clear after processing
        await self._session.flush()

    async def mark_error(self, job: AnalysisJob, error: str) -> None:
        """Завершить job с ошибкой. error обрезается до 1000 символов."""
        job.status = JobStatus.ERROR
        job.error_message = (error or "")[:1000]
        job.finished_at = _utcnow()
        job.input_text = None
        await self._session.flush()

    async def recover_stale_processing(self, timeout_minutes: int = 10) -> int:
        """Сбросить зависшие PROCESSING обратно в PENDING.

        Сценарий: worker умер между `claim_next` и `mark_success/_error`.
        Job со `started_at < NOW() - timeout` считаем брошенным.
        Возвращает число восстановленных записей.
        """
        cutoff = _utcnow() - timedelta(minutes=timeout_minutes)
        stmt = (
            update(AnalysisJob)
            .where(
                AnalysisJob.status == JobStatus.PROCESSING,
                AnalysisJob.started_at < cutoff,
            )
            .values(status=JobStatus.PENDING, worker_id=None, started_at=None)
        )
        result = await self._session.execute(stmt)
        return result.rowcount or 0


class BatchJobRepository:
    """Batch-job операции, общие для роутеров и worker'а."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_owned(self, batch_id: UUID, user_id: UUID) -> BatchJob | None:
        """Вернуть batch, если он принадлежит юзеру; иначе None.

        Чужие batch'и отдаются как 404 на уровне роутера — не утекаем факт
        существования через 403.
        """
        stmt = select(BatchJob).where(BatchJob.id == batch_id, BatchJob.user_id == user_id)
        return (await self._session.execute(stmt)).scalar_one_or_none()


class DeploymentRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def save(self, deployment: ModelDeployment) -> None:
        self._session.add(deployment)

    async def get_latest(self) -> ModelDeployment | None:
        stmt = select(ModelDeployment).order_by(ModelDeployment.deployed_at.desc()).limit(1)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()


class UserRepository:
    """CRUD по таблице users. Email хранится как введён, ищется через LOWER()."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, user: User) -> User:
        """Insert нового пользователя; UNIQUE-конфликт по email пробрасывается наверх."""
        self._session.add(user)
        await self._session.flush()
        return user

    async def get_by_id(self, user_id: UUID | str) -> User | None:
        stmt = select(User).where(User.id == user_id)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def get_by_email(self, email: str) -> User | None:
        # Регистронезависимый поиск — стандартное поведение для email.
        stmt = select(User).where(func.lower(User.email) == email.strip().lower())
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def update_last_login(self, user: User) -> None:
        user.last_login_at = _utcnow()
        await self._session.flush()


class RefreshTokenRepository:
    """Refresh-токены: хранятся sha256 хеши, поддерживается rotation + reuse-detection.

    Reuse-detection: если в `refresh` пришёл уже revoked-токен — это украденный токен,
    помечаем всю активную цепь токенов пользователя как revoked (см. revoke_all_for_user).
    """

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, token: RefreshToken) -> RefreshToken:
        self._session.add(token)
        await self._session.flush()
        return token

    async def get_by_token_hash(self, token_hash: str) -> RefreshToken | None:
        stmt = select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def revoke(self, token: RefreshToken, replaced_by: RefreshToken | None = None) -> None:
        token.revoked = True
        token.revoked_at = _utcnow()
        if replaced_by is not None:
            token.replaced_by_id = replaced_by.id
        await self._session.flush()

    async def revoke_all_for_user(self, user_id: UUID | str) -> int:
        """Инвалидировать все активные токены пользователя. Возвращает кол-во revoked.

        Используется при reuse-detection (украденный токен) и в admin-flow.
        """
        stmt = select(RefreshToken).where(
            RefreshToken.user_id == user_id,
            RefreshToken.revoked.is_(False),
        )
        rows = (await self._session.execute(stmt)).scalars().all()
        now = _utcnow()
        for row in rows:
            row.revoked = True
            row.revoked_at = now
        await self._session.flush()
        return len(rows)
