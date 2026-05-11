"""Тесты JobRepository — async-job очередь поверх PostgreSQL.

Использует реальный Postgres через `db_session` fixture (conftest.py).
"""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import text as sql_text

from app.database.models import AnalysisJob, JobStatus
from app.database.repositories import JobRepository

pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest.fixture(autouse=True)
async def _cleanup_jobs(db_session):
    """analysis_jobs делится между тестами, claim_next берёт ЛЮБОЙ PENDING.
    Поэтому начинаем каждый тест с пустой таблицы (тесты independent)."""
    await db_session.execute(sql_text("DELETE FROM analysis_jobs"))
    await db_session.commit()
    yield
    await db_session.execute(sql_text("DELETE FROM analysis_jobs"))
    await db_session.commit()


def _make_job(text: str = "тестовый текст 123", explain: bool = False) -> AnalysisJob:
    """Build an AnalysisJob, готовый к insert."""
    from uuid_utils import uuid7

    return AnalysisJob(
        id=uuid7(),
        request_id=uuid7(),
        input_text=text,
        text_hash="a" * 64,
        text_length=len(text),
        detector_type="cascade",
        explain=explain,
        status=JobStatus.PENDING,
    )


class TestCreate:
    async def test_inserts_pending_job(self, db_session):
        repo = JobRepository(db_session)
        job = _make_job()
        saved = await repo.create(job)
        await db_session.commit()

        assert str(saved.id) == str(job.id)
        assert saved.status == JobStatus.PENDING
        assert saved.input_text == job.input_text
        assert saved.detector_type == "cascade"

    async def test_idempotent_on_duplicate_request_id(self, db_session):
        """Повторный create с тем же request_id возвращает существующий job (не падает)."""
        repo = JobRepository(db_session)
        job1 = _make_job(text="первый текст")
        first = await repo.create(job1)
        await db_session.commit()

        # Создаём «другой» job с тем же request_id (идемпотентный POST)
        job2 = _make_job(text="второй текст")
        job2.request_id = job1.request_id
        second = await repo.create(job2)
        await db_session.commit()

        # Возвращается first (тот, что вставлен), а не job2
        assert str(second.id) == str(first.id)
        assert second.input_text == job1.input_text  # текст первого, не второго


class TestGet:
    async def test_get_by_id_returns_job(self, db_session):
        repo = JobRepository(db_session)
        job = _make_job()
        await repo.create(job)
        await db_session.commit()

        fetched = await repo.get_by_id(job.id)
        assert fetched is not None
        assert str(fetched.id) == str(job.id)

    async def test_get_by_id_missing_returns_none(self, db_session):
        repo = JobRepository(db_session)
        result = await repo.get_by_id(str(uuid4()))
        assert result is None

    async def test_get_by_request_id(self, db_session):
        repo = JobRepository(db_session)
        job = _make_job()
        await repo.create(job)
        await db_session.commit()

        fetched = await repo.get_by_request_id(job.request_id)
        assert fetched is not None
        assert str(fetched.id) == str(job.id)


class TestClaimNext:
    async def test_claim_pending_marks_processing(self, db_session):
        repo = JobRepository(db_session)
        job = _make_job()
        await repo.create(job)
        await db_session.commit()

        claimed = await repo.claim_next("worker-1")
        await db_session.commit()

        assert claimed is not None
        assert str(claimed.id) == str(job.id)
        assert claimed.status == JobStatus.PROCESSING
        assert claimed.worker_id == "worker-1"
        assert claimed.started_at is not None

    async def test_claim_next_when_empty_returns_none(self, db_session):
        repo = JobRepository(db_session)
        result = await repo.claim_next("worker-1")
        assert result is None

    async def test_claim_skips_already_processing(self, db_session):
        """Job уже взят первым worker — второй не должен получить тот же."""
        repo = JobRepository(db_session)
        job = _make_job()
        await repo.create(job)
        await db_session.commit()

        first = await repo.claim_next("worker-1")
        await db_session.commit()
        assert first is not None

        second = await repo.claim_next("worker-2")
        await db_session.commit()
        assert second is None  # больше нет PENDING

    async def test_claims_oldest_first(self, db_session):
        """FIFO порядок по created_at."""
        repo = JobRepository(db_session)
        old = _make_job(text="старый")
        new = _make_job(text="новый")
        await repo.create(old)
        await db_session.commit()
        # await небольшую паузу чтобы created_at отличался
        await db_session.execute(sql_text("SELECT pg_sleep(0.01)"))
        await repo.create(new)
        await db_session.commit()

        first = await repo.claim_next("worker-1")
        await db_session.commit()
        assert first is not None
        assert str(first.id) == str(old.id)  # старый job взят первым


class TestMarkers:
    async def test_mark_success_clears_input_text(self, db_session):
        """Privacy: после SUCCESS input_text обнуляется. analysis_id ссылается на FK."""
        from datetime import datetime, timezone

        from app.database.models import Analysis
        from app.database.repositories import AnalysisRepository

        # Создаём фейковый Analysis для FK
        analysis = Analysis(
            id=uuid4(),
            request_id=uuid4(),
            text_hash="b" * 64,
            text_length=100,
            status="SUCCESS",
            verdict="ai",
            confidence=0.9,
            risk_level="HIGH",
            detector_used="cascade",
            inference_ms=42.0,
            cached=False,
            model_version="test",
            service_version="0.1.0",
            warnings=[],
            requested_at=datetime.now(timezone.utc),
        )
        await AnalysisRepository(db_session).save(analysis)
        await db_session.commit()

        repo = JobRepository(db_session)
        job = _make_job(text="секретный текст пользователя")
        await repo.create(job)
        await db_session.commit()

        claimed = await repo.claim_next("worker-1")
        await db_session.commit()
        assert claimed is not None

        await repo.mark_success(claimed, analysis_id=analysis.id)
        await db_session.commit()

        fresh = await repo.get_by_id(job.id)
        assert fresh is not None
        assert fresh.status == JobStatus.SUCCESS
        assert fresh.input_text is None  # privacy
        assert fresh.finished_at is not None
        assert str(fresh.analysis_id) == str(analysis.id)

    async def test_mark_error_clears_input_text_and_truncates(self, db_session):
        """Error: input_text обнуляется, error_message обрезается до 1000."""
        repo = JobRepository(db_session)
        job = _make_job()
        await repo.create(job)
        await db_session.commit()

        claimed = await repo.claim_next("worker-1")
        await db_session.commit()
        assert claimed is not None

        long_error = "x" * 2000
        await repo.mark_error(claimed, error=long_error)
        await db_session.commit()

        fresh = await repo.get_by_id(job.id)
        assert fresh is not None
        assert fresh.status == JobStatus.ERROR
        assert fresh.input_text is None
        assert fresh.error_message is not None
        assert len(fresh.error_message) == 1000


class TestRecoverStaleProcessing:
    """Восстановление job'ов, у которых worker умер до mark_success/_error."""

    async def test_old_processing_returned_to_pending(self, db_session):
        repo = JobRepository(db_session)
        job = _make_job()
        await repo.create(job)
        await db_session.commit()

        # Имитация: worker подобрал job 20 минут назад и больше не отозвался.
        claimed = await repo.claim_next("dead-worker")
        assert claimed is not None
        claimed.started_at = datetime.now(timezone.utc) - timedelta(minutes=20)
        await db_session.commit()

        recovered = await repo.recover_stale_processing(timeout_minutes=10)
        await db_session.commit()

        assert recovered == 1
        fresh = await repo.get_by_id(job.id)
        assert fresh is not None
        assert fresh.status == JobStatus.PENDING
        assert fresh.worker_id is None
        assert fresh.started_at is None

    async def test_fresh_processing_not_touched(self, db_session):
        repo = JobRepository(db_session)
        job = _make_job()
        await repo.create(job)
        await db_session.commit()

        claimed = await repo.claim_next("active-worker")
        assert claimed is not None
        await db_session.commit()

        recovered = await repo.recover_stale_processing(timeout_minutes=10)
        await db_session.commit()

        assert recovered == 0
        fresh = await repo.get_by_id(job.id)
        assert fresh is not None
        assert fresh.status == JobStatus.PROCESSING
        assert fresh.worker_id == "active-worker"

    async def test_pending_jobs_not_touched(self, db_session):
        """PENDING не должен меняться recover'ом."""
        repo = JobRepository(db_session)
        job = _make_job()
        await repo.create(job)
        await db_session.commit()

        recovered = await repo.recover_stale_processing(timeout_minutes=10)
        await db_session.commit()

        assert recovered == 0
        fresh = await repo.get_by_id(job.id)
        assert fresh is not None
        assert fresh.status == JobStatus.PENDING
