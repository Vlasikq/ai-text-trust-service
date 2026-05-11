"""Воркер для асинхронной обработки каскадных задач.

Запуск локально: `uv run python -m app.worker`. В Docker — той же командой
внутри образа `aitrust-api`.

Алгоритм:
  1. Инициализация engine и детектора (по умолчанию каскад с трансформером).
  2. В цикле: `claim_next` через `FOR UPDATE SKIP LOCKED`, вызов `run_detection`,
     сохранение `Analysis`, перевод задачи в `SUCCESS`.
  3. По сигналу `SIGTERM` дорабатываем текущий job, новых не берём, освобождаем ресурсы.

Инвариант: код анализа здесь тот же, что в `/api/v1/analyze`, и приходит через
`services/detection.py`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import sys
from datetime import datetime, timezone

from sqlalchemy import case, func, update

from app.cache import DetectionCache
from app.calibration.platt import ProbabilityCalibrator
from app.config import Settings, get_settings
from app.database.engine import dispose_engine, init_engine
from app.database.models import BatchJob, BatchStatus
from app.database.persist import response_to_model
from app.database.uow import UnitOfWork
from app.detectors.factory import build_detector
from app.explanation.markers import StyleExplainer
from app.logging_config import set_correlation_id, setup_logging
from app.preprocessing.pipeline import TextPreprocessor
from app.schemas import Status
from app.services.detection import DetectionContext, run_detection

log = logging.getLogger(__name__)

WORKER_ID = f"worker-{socket.gethostname()}-{os.getpid()}"
POLL_INTERVAL_S = 2.0
ERROR_BACKOFF_S = 5.0

_shutdown_requested = False


def _build_context(settings: Settings) -> DetectionContext:
    detector = build_detector(settings)
    log.info("Worker detector ready: %s", detector.get_info())

    calibrator: ProbabilityCalibrator | None = None
    try:
        calibrator = ProbabilityCalibrator(settings.artifacts_dir / "calibration")
        calibrator.load()
    except Exception:
        log.warning("Calibrator not loaded — raw probabilities used")
        calibrator = None

    explainer: StyleExplainer | None = None
    try:
        baselines_path = settings.artifacts_dir / "human_baselines.json"
        explainer = StyleExplainer(baselines_path)
        explainer.load()
        if not explainer.is_ready():
            explainer = None
    except Exception:
        # Без exc_info оператор видел бы только «explainer = None» и не понимал,
        # что упало: отсутствующий baselines.json, mismatched schema или OOM.
        log.warning("Explainer load failed in worker context, continuing without it", exc_info=True)
        explainer = None

    return DetectionContext(
        settings=settings,
        preprocessor=TextPreprocessor(settings),
        detector=detector,
        calibrator=calibrator,
        explainer=explainer,
        cache=DetectionCache(maxsize=settings.cache_maxsize),
    )


async def _bump_batch_progress(session, batch_id: str, *, success: bool) -> None:
    """Атомарно инкрементить счётчик batch'а после завершения child-job.

    Делаем через SQL-expression `n_completed = n_completed + 1`, а не через
    ORM-чтение → Python-инкремент → запись. Это защищает от race condition
    при `--scale worker=2`: оба воркера могут обрабатывать разные child-jobs
    одного batch'а одновременно, и read-modify-write потерял бы один инкремент.

    Когда суммарный counter достигает `n_total` — переводим batch в COMPLETED
    и проставляем `completed_at`. Условие в `case()` оперирует уже инкрементом
    ТЕКУЩЕЙ операции (`n_completed + 1`), потому что VALUES вычисляются
    одновременно — иначе мы сначала бы записали инкремент, а потом проверили,
    но конструкция UPDATE атомарна.
    """
    if success:
        new_completed = BatchJob.n_completed + 1
        new_failed = BatchJob.n_failed
    else:
        new_completed = BatchJob.n_completed
        new_failed = BatchJob.n_failed + 1
    is_last = (new_completed + new_failed) == BatchJob.n_total

    await session.execute(
        update(BatchJob)
        .where(BatchJob.id == batch_id)
        .values(
            n_completed=new_completed,
            n_failed=new_failed,
            status=case(
                (is_last, BatchStatus.COMPLETED),
                else_=BatchStatus.PROCESSING,
            ),
            # Если ещё не последний — оставляем как было (NULL до этого).
            completed_at=case(
                (is_last, func.now()),
                else_=BatchJob.completed_at,
            ),
        )
    )


async def process_one_job(ctx: DetectionContext, deployment_id: str | None) -> bool:
    """Взять и обработать один job. Возвращает True если job обработан.

    Public — используется из e2e-тестов batch'а (inline без поднятия процесса).
    """
    async with UnitOfWork() as uow:
        job = await uow.jobs.claim_next(WORKER_ID)
        if job is None:
            await uow.commit()  # release tx
            return False

        request_id = str(job.request_id)
        batch_id = str(job.batch_id) if job.batch_id else None
        set_correlation_id(request_id)
        text = job.input_text or ""
        log.info(
            "job_claimed",
            extra={
                "job_id": str(job.id),
                "text_length": job.text_length,
                "explain": job.explain,
                "batch_id": batch_id,
            },
        )

        try:
            outcome = await run_detection(
                ctx,
                text,
                request_id=request_id,
                explain=job.explain,
                debug=False,
                timeout_s=None,  # без таймаута: cascade в серой зоне на CPU = 10+ сек
            )

            success = (
                outcome.detection is not None
                and outcome.response.status == Status.success
            )

            if success:
                analysis = response_to_model(
                    response=outcome.response,
                    cleaned_text=outcome.preprocessing.text,
                    detector_method=outcome.detection.method,
                    cascade_path=outcome.detection.metadata.get("cascade_path"),
                    inference_ms=outcome.detection.inference_ms,
                    cached=outcome.cached_hit,
                    model_version=ctx.settings.model_version,
                    service_version=ctx.settings.service_version,
                    deployment_id=deployment_id,
                    user_id=str(job.user_id) if job.user_id else None,
                )
                await uow.analyses.save(analysis)
                await uow.jobs.mark_success(job, analysis_id=analysis.id)
                verdict_val = (
                    outcome.response.verdict.value if outcome.response.verdict else None
                )
                log.info(
                    "job_succeeded",
                    extra={
                        "job_id": str(job.id),
                        "verdict": verdict_val,
                        "confidence": outcome.response.confidence,
                        "cascade_path": outcome.detection.metadata.get("cascade_path"),
                    },
                )
            else:
                # NO_DECISION или ERROR (timeout) — Analysis не сохраняем
                warn_codes = [w.value for w in outcome.response.warnings]
                err = f"status={outcome.response.status.value}; warnings={warn_codes}"
                await uow.jobs.mark_error(job, error=err)
                log.warning(
                    "job_no_decision_or_error",
                    extra={"job_id": str(job.id), "reason": err},
                )

            if batch_id is not None:
                await _bump_batch_progress(uow.session, batch_id, success=success)

            await uow.commit()
            return True

        except Exception as e:  # broad except: worker должен пережить любую ошибку и пометить job
            log.exception("job_processing_failed", extra={"job_id": str(job.id)})
            await uow.jobs.mark_error(job, error=f"{type(e).__name__}: {e}")
            if batch_id is not None:
                await _bump_batch_progress(uow.session, batch_id, success=False)
            await uow.commit()
            return True
        finally:
            set_correlation_id(None)


# Обратная совместимость: старый импорт `_process_one_job` где-то мог сохраниться.
_process_one_job = process_one_job


async def main_loop(ctx: DetectionContext, deployment_id: str | None) -> None:
    log.info(
        "worker_loop_start",
        extra={"worker_id": WORKER_ID, "poll_interval_s": POLL_INTERVAL_S},
    )
    while not _shutdown_requested:
        try:
            processed = await process_one_job(ctx, deployment_id)
            if not processed:
                await asyncio.sleep(POLL_INTERVAL_S)
        except Exception:
            log.exception("worker_loop_iteration_failed")
            await asyncio.sleep(ERROR_BACKOFF_S)
    log.info("worker_loop_stopped")


def _install_signal_handlers() -> None:
    def _handle(sig, _frame):
        global _shutdown_requested
        log.info("signal_received", extra={"signal": sig})
        _shutdown_requested = True

    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, _handle)
        signal.signal(signal.SIGINT, _handle)


async def _record_deployment(settings: Settings) -> str | None:
    """Регистрирует worker как ModelDeployment (как api в main.py)."""
    from app.database.models import ModelDeployment

    try:
        async with UnitOfWork() as uow:
            d = ModelDeployment(
                model_version=settings.model_version,
                service_version=f"{settings.service_version}-worker",
                detector_type=settings.detector_type.value,
                config_snapshot={
                    "cascade_lo": settings.cascade_lo,
                    "cascade_hi": settings.cascade_hi,
                    "worker_id": WORKER_ID,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            await uow.deployments.save(d)
            await uow.commit()
            return str(d.id)
    except Exception:
        log.warning("Worker deployment record failed", exc_info=True)
        return None


async def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    log.info("worker_starting", extra={"worker_id": WORKER_ID})
    _install_signal_handlers()

    init_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_pool_overflow,
    )

    deployment_id = await _record_deployment(settings)

    # Восстановление зависших job'ов: предыдущий инстанс мог умереть между
    # claim_next и mark_*, иначе такие job'ы остаются в PROCESSING навсегда.
    try:
        async with UnitOfWork() as uow:
            recovered = await uow.jobs.recover_stale_processing(timeout_minutes=10)
            await uow.commit()
        if recovered:
            log.warning(
                "stale_jobs_recovered",
                extra={"count": recovered, "timeout_minutes": 10},
            )
    except Exception:
        log.warning("stale_recovery_failed", exc_info=True)

    ctx = _build_context(settings)

    # Warmup
    try:
        ctx.detector.predict("Тестовый текст для прогрева воркера.")
        log.info("worker_warmup_complete")
    except Exception:
        log.warning("worker_warmup_failed", exc_info=True)

    try:
        await main_loop(ctx, deployment_id)
    finally:
        await dispose_engine()
        log.info("worker_shutdown_complete")


if __name__ == "__main__":
    asyncio.run(main())
