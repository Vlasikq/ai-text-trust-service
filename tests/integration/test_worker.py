"""Тесты src/app/worker.py — async-jobs процессор.

Покрытие:
  - process_one_job: happy path, no-pending, NO_DECISION (text_too_short),
    detector exception, privacy (input_text занулён).
  - _bump_batch_progress: атомарный инкремент счётчиков, переход в COMPLETED
    при последнем child, корректные status/completed_at.

E2E через `POST /api/v1/batch` лежит в test_batch_api.py — здесь фокус на
worker-примитивах без HTTP-слоя, потому что:
  1. claim_next через FOR UPDATE SKIP LOCKED — критичный invariant очереди;
  2. _bump_batch_progress использует SQL-expression update, race-safe только
     при реальной БД.
"""

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy import text as sql_text
from uuid_utils import uuid7

from app.cache import DetectionCache
from app.config import Settings
from app.database import engine as engine_mod
from app.database.models import (
    AnalysisJob,
    BatchJob,
    BatchStatus,
    JobStatus,
    User,
)
from app.database.uow import UnitOfWork
from app.preprocessing.pipeline import TextPreprocessor
from app.services.detection import DetectionContext
from app.worker import _bump_batch_progress, process_one_job
from tests.conftest import FakeDetector

pytestmark = pytest.mark.asyncio(loop_scope="session")


# ── Fixtures ──────────────────────────────────────────────────


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def worker_engine(postgres_url, _engine):
    """Engine для worker-тестов. Таблицы уже созданы фикстурой _engine."""
    engine_mod.init_engine(postgres_url, pool_size=2, max_overflow=2)
    yield
    await engine_mod.dispose_engine()


@pytest_asyncio.fixture(autouse=True, loop_scope="session")
async def _wipe_tables(worker_engine):
    """Чистим job-related + users перед/после каждого теста.

    Порядок: analysis_jobs (FK на analyses + users + batch_jobs), analyses
    (FK на users SET NULL), batch_jobs (FK на users CASCADE), users.
    """
    async with engine_mod.async_session() as s:
        for table in ("analysis_jobs", "analyses", "batch_jobs", "refresh_tokens", "users"):
            await s.execute(sql_text(f"DELETE FROM {table}"))
        await s.commit()
    yield


# ── Helpers ───────────────────────────────────────────────────


def _ctx(detector=None, *, min_chars: int = 10, max_chars: int = 8000) -> DetectionContext:
    """Сборка тестового DetectionContext: settings → preprocessor → детектор.

    По дефолту FakeDetector(prob_ai=0.85) — verdict=ai, risk=HIGH.
    """
    settings = Settings(min_chars=min_chars, max_chars=max_chars)
    return DetectionContext(
        settings=settings,
        preprocessor=TextPreprocessor(settings),
        detector=detector if detector is not None else FakeDetector(prob_ai=0.85),
        calibrator=None,
        explainer=None,
        cache=DetectionCache(maxsize=128),
    )


async def _insert_pending_job(
    *,
    text: str = "Длинный текст для анализа воркером — заведомо проходит min_chars=10.",
    detector_type: str = "cascade",
    explain: bool = False,
    batch_id: str | None = None,
    user_id: str | None = None,
) -> AnalysisJob:
    """Вставить PENDING job; вернуть свежезагруженный объект (с id).

    JobRepository.create() намеренно не пропагирует batch_id / batch_external_*
    в pg_insert (в проде batch-роутер делает bulk-insert через session.execute
    с полным dict). Здесь патчим batch_id отдельным UPDATE через ORM-tracked
    мутацию — это репродуцирует состояние «child job привязан к batch».
    """
    async with UnitOfWork() as uow:
        job = AnalysisJob(
            id=uuid7(),
            request_id=uuid7(),
            input_text=text,
            text_hash="a" * 64,
            text_length=len(text),
            detector_type=detector_type,
            explain=explain,
            status=JobStatus.PENDING,
            user_id=user_id,
        )
        created = await uow.jobs.create(job)
        if batch_id is not None:
            created.batch_id = batch_id  # ORM-tracked → UPDATE на commit
        await uow.commit()
    return created


async def _insert_user_and_batch(*, n_total: int = 2) -> tuple[str, str]:
    """Создать User + BatchJob (status=PENDING), вернуть (user_id, batch_id)."""
    user_id = uuid7()
    batch_id = uuid7()
    async with UnitOfWork() as uow:
        await uow.users.create(
            User(
                id=user_id,
                email=f"u-{user_id}@example.com",
                password_hash="$argon2id$dummy",
            )
        )
        uow.session.add(
            BatchJob(
                id=batch_id,
                user_id=user_id,
                n_total=n_total,
                status=BatchStatus.PENDING,
            )
        )
        await uow.commit()
    return str(user_id), str(batch_id)


async def _get_job(job_id) -> AnalysisJob:
    async with engine_mod.async_session() as s:
        return (
            await s.execute(select(AnalysisJob).where(AnalysisJob.id == job_id))
        ).scalar_one()


async def _get_batch(batch_id) -> BatchJob:
    async with engine_mod.async_session() as s:
        return (
            await s.execute(select(BatchJob).where(BatchJob.id == batch_id))
        ).scalar_one()


# ── process_one_job ───────────────────────────────────────────


class TestProcessOneJob:
    async def test_happy_path_creates_analysis_and_clears_input(self):
        """PENDING job → SUCCESS, Analysis сохранён, input_text занулён (privacy)."""
        job = await _insert_pending_job()
        processed = await process_one_job(_ctx(), deployment_id=None)
        assert processed is True

        updated = await _get_job(job.id)
        assert updated.status == JobStatus.SUCCESS
        assert updated.analysis_id is not None
        assert updated.started_at is not None
        assert updated.finished_at is not None
        assert updated.worker_id is not None
        # Privacy invariant: input_text должен быть NULL после завершения.
        assert updated.input_text is None

    async def test_no_pending_returns_false(self):
        """Когда очередь пуста — возвращаем False, не упадаем."""
        processed = await process_one_job(_ctx(), deployment_id=None)
        assert processed is False

    async def test_text_too_short_marks_error_no_decision(self):
        """Препроцессор отдаёт NO_DECISION (TEXT_TOO_SHORT) → job помечен ERROR.

        Analysis не сохраняется. input_text всё равно занулён.
        """
        await _insert_pending_job(text="Короткий")
        ctx = _ctx(min_chars=300)  # текст 8 символов, не проходит min_chars
        processed = await process_one_job(ctx, deployment_id=None)
        assert processed is True

        async with engine_mod.async_session() as s:
            jobs = (await s.execute(select(AnalysisJob))).scalars().all()
        assert len(jobs) == 1
        assert jobs[0].status == JobStatus.ERROR
        assert jobs[0].analysis_id is None
        assert "TEXT_TOO_SHORT" in (jobs[0].error_message or "")
        assert jobs[0].input_text is None

    async def test_detector_exception_marks_error(self):
        """Исключение в detector.predict → job=ERROR с типом+сообщением исключения."""
        await _insert_pending_job()
        ctx = _ctx(detector=FakeDetector(raises=RuntimeError("synthetic-failure")))
        processed = await process_one_job(ctx, deployment_id=None)
        assert processed is True  # job обработан (с ошибкой), не упал worker-loop

        async with engine_mod.async_session() as s:
            jobs = (await s.execute(select(AnalysisJob))).scalars().all()
        assert len(jobs) == 1
        assert jobs[0].status == JobStatus.ERROR
        err = jobs[0].error_message or ""
        assert "RuntimeError" in err
        assert "synthetic-failure" in err
        assert jobs[0].input_text is None

    async def test_processed_job_no_longer_claimed(self):
        """После обработки следующий process_one_job не подхватит тот же job."""
        await _insert_pending_job()
        await process_one_job(_ctx(), deployment_id=None)
        # Очередь PENDING пуста — следующий вызов вернёт False.
        again = await process_one_job(_ctx(), deployment_id=None)
        assert again is False

    async def test_two_jobs_processed_sequentially(self):
        """Два PENDING jobs → два последовательных вызова обработают оба."""
        await _insert_pending_job()
        await _insert_pending_job(text="Второй длинный текст для воркера.")

        assert await process_one_job(_ctx(), deployment_id=None) is True
        assert await process_one_job(_ctx(), deployment_id=None) is True
        assert await process_one_job(_ctx(), deployment_id=None) is False

        async with engine_mod.async_session() as s:
            jobs = (await s.execute(select(AnalysisJob))).scalars().all()
        assert len(jobs) == 2
        assert all(j.status == JobStatus.SUCCESS for j in jobs)


# ── _bump_batch_progress ──────────────────────────────────────


class TestBumpBatchProgress:
    async def test_success_increments_n_completed_and_sets_processing(self):
        """Первый успешный child → n_completed=1, status=PROCESSING, completed_at NULL."""
        _, batch_id = await _insert_user_and_batch(n_total=3)

        async with UnitOfWork() as uow:
            await _bump_batch_progress(uow.session, batch_id, success=True)
            await uow.commit()

        batch = await _get_batch(batch_id)
        assert batch.n_completed == 1
        assert batch.n_failed == 0
        assert batch.status == BatchStatus.PROCESSING
        assert batch.completed_at is None

    async def test_failure_increments_n_failed(self):
        """Failed child → n_failed=1, n_completed=0, status=PROCESSING."""
        _, batch_id = await _insert_user_and_batch(n_total=2)

        async with UnitOfWork() as uow:
            await _bump_batch_progress(uow.session, batch_id, success=False)
            await uow.commit()

        batch = await _get_batch(batch_id)
        assert batch.n_completed == 0
        assert batch.n_failed == 1
        assert batch.status == BatchStatus.PROCESSING

    async def test_last_child_transitions_to_completed(self):
        """Когда (n_completed + n_failed) == n_total → status=COMPLETED + completed_at."""
        _, batch_id = await _insert_user_and_batch(n_total=2)

        async with UnitOfWork() as uow:
            await _bump_batch_progress(uow.session, batch_id, success=True)
            await uow.commit()
        async with UnitOfWork() as uow:
            await _bump_batch_progress(uow.session, batch_id, success=True)
            await uow.commit()

        batch = await _get_batch(batch_id)
        assert batch.status == BatchStatus.COMPLETED
        assert batch.n_completed == 2
        assert batch.n_failed == 0
        assert batch.completed_at is not None

    async def test_mixed_outcomes_still_complete(self):
        """1 success + 1 failure (n_total=2) → status=COMPLETED, счётчики разделены."""
        _, batch_id = await _insert_user_and_batch(n_total=2)

        async with UnitOfWork() as uow:
            await _bump_batch_progress(uow.session, batch_id, success=True)
            await uow.commit()
        async with UnitOfWork() as uow:
            await _bump_batch_progress(uow.session, batch_id, success=False)
            await uow.commit()

        batch = await _get_batch(batch_id)
        assert batch.status == BatchStatus.COMPLETED
        assert batch.n_completed == 1
        assert batch.n_failed == 1
        assert batch.completed_at is not None


# ── process_one_job в связке с batch ─────────────────────────


class TestProcessOneJobWithBatch:
    async def test_completes_batch_counters(self):
        """Полный flow: 2 child jobs одного batch → process_one_job дважды →
        BatchJob переходит в COMPLETED, счётчик n_completed=2."""
        user_id, batch_id = await _insert_user_and_batch(n_total=2)
        await _insert_pending_job(batch_id=batch_id, user_id=user_id)
        await _insert_pending_job(
            batch_id=batch_id, user_id=user_id,
            text="Второй текст для batch-обработки воркером.",
        )

        assert await process_one_job(_ctx(), deployment_id=None) is True
        # После первого: batch=PROCESSING, n_completed=1
        batch = await _get_batch(batch_id)
        assert batch.status == BatchStatus.PROCESSING
        assert batch.n_completed == 1

        assert await process_one_job(_ctx(), deployment_id=None) is True
        # После второго (последнего): batch=COMPLETED
        batch = await _get_batch(batch_id)
        assert batch.status == BatchStatus.COMPLETED
        assert batch.n_completed == 2
        assert batch.completed_at is not None
