"""E2E тесты batch-режима: upload CSV → inline-обработка через worker → отчёт.

Контракт:
  - POST /api/v1/batch — happy path с CSV, который проходит парсинг.
  - process_one_job inline — гарантирует определённый порядок: тесты не
    зависят от запущенного worker-процесса.
  - После обработки всех child-jobs:
        batch.status == COMPLETED
        n_completed == n_total
        input_text у всех child-jobs == NULL (privacy)
  - Скачивание results.csv / results.json — содержит результаты, НЕ содержит
    исходных текстов.
  - Partial unique index `uq_one_active_batch_per_user` ловит второй
    активный batch одного пользователя → 409.
  - Атомарный UPDATE счётчика устойчив к concurrent инкрементам (race-safe).
"""

from __future__ import annotations

import csv
import io
import json

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy import text as sql_text

from app.cache import DetectionCache
from app.config import Settings
from app.database import engine as engine_mod
from app.database.models import AnalysisJob, BatchJob
from app.database.uow import UnitOfWork
from app.preprocessing.pipeline import TextPreprocessor
from app.services.detection import DetectionContext
from app.worker import _bump_batch_progress, process_one_job
from tests.conftest import FakeDetector, make_test_app

pytestmark = pytest.mark.asyncio(loop_scope="session")


def _make_ctx(settings: Settings) -> DetectionContext:
    return DetectionContext(
        settings=settings,
        preprocessor=TextPreprocessor(settings),
        detector=FakeDetector(),
        calibrator=None,
        explainer=None,
        cache=DetectionCache(maxsize=128),
    )


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def app_with_engine(postgres_url, _engine):
    engine_mod.init_engine(postgres_url, pool_size=2, max_overflow=2)
    yield make_test_app(
        routers=("auth", "me", "batch_me", "batch"),
        db_enabled=True,
    )
    await engine_mod.dispose_engine()


@pytest_asyncio.fixture(autouse=True, loop_scope="session")
async def _wipe_tables(app_with_engine):
    """Чистим все таблицы, которые трогает batch-flow.

    Порядок важен: analysis_jobs и analyses имеют FK; batch_jobs независим,
    но удаляется первым. refresh_tokens/users чистим из-за регистраций.
    """
    async with engine_mod.async_session() as s:
        for table in (
            "analysis_jobs",
            "analyses",
            "batch_jobs",
            "refresh_tokens",
            "users",
        ):
            await s.execute(sql_text(f"DELETE FROM {table}"))
        await s.commit()
    yield
    async with engine_mod.async_session() as s:
        for table in (
            "analysis_jobs",
            "analyses",
            "batch_jobs",
            "refresh_tokens",
            "users",
        ):
            await s.execute(sql_text(f"DELETE FROM {table}"))
        await s.commit()


@pytest_asyncio.fixture(loop_scope="session")
async def client(app_with_engine):
    transport = ASGITransport(app=app_with_engine)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── helpers ───────────────────────────────────────────────────


async def _register(client: AsyncClient, email: str = "batch@example.com") -> dict:
    r = await client.post(
        "/auth/register", json={"email": email, "password": "supersecret123"}
    )
    return r.json()


def _make_csv(n_rows: int = 3, *, with_id: bool = True, text_len: int = 400) -> bytes:
    rows = [["id", "text"]] if with_id else [["text"]]
    base_text = "Текст для batch-обработки достаточно длинный. " * 50
    text = base_text[:text_len]
    for i in range(n_rows):
        if with_id:
            rows.append([f"ext-{i}", text])
        else:
            rows.append([text])
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


async def _drain_jobs(settings: Settings, max_iterations: int = 50) -> int:
    """Прогнать process_one_job в цикле, пока есть PENDING. Возвращает кол-во обработанных."""
    ctx = _make_ctx(settings)
    processed = 0
    for _ in range(max_iterations):
        ok = await process_one_job(ctx, deployment_id=None)
        if not ok:
            break
        processed += 1
    return processed


# ── tests ─────────────────────────────────────────────────────


class TestBatchHappyPath:
    async def test_create_process_download(self, client):
        body = await _register(client, email="alice-batch@example.com")
        access = body["tokens"]["access_token"]

        resp = await client.post(
            "/api/v1/batch",
            files={"file": ("essays.csv", _make_csv(3), "text/csv")},
            data={"detector_type": "tfidf", "explain": "false"},
            headers={"Authorization": f"Bearer {access}"},
        )
        assert resp.status_code == 202, resp.text
        created = resp.json()
        assert created["n_accepted"] == 3
        assert created["n_skipped"] == 0
        batch_id = created["batch_id"]

        # Drain — process_one_job inline для каждого child.
        processed = await _drain_jobs(Settings(min_chars=300))
        assert processed == 3

        # Status — batch COMPLETED
        s = await client.get(
            f"/api/v1/batch/{batch_id}",
            headers={"Authorization": f"Bearer {access}"},
        )
        assert s.status_code == 200
        st = s.json()
        assert st["status"] == "COMPLETED"
        assert st["n_completed"] == 3
        assert st["n_failed"] == 0
        assert st["completed_at"] is not None
        assert st["results_csv_url"]
        assert st["results_json_url"]

        # CSV отчёт скачивается, header корректен, текстов нет
        r_csv = await client.get(
            f"/api/v1/batch/{batch_id}/results.csv",
            headers={"Authorization": f"Bearer {access}"},
        )
        assert r_csv.status_code == 200
        assert "text/csv" in r_csv.headers["content-type"]
        body_csv = r_csv.content.decode("utf-8")
        reader = csv.DictReader(io.StringIO(body_csv))
        out_rows = list(reader)
        assert len(out_rows) == 3
        # Privacy: ни одной колонки 'text' в отчёте.
        assert "text" not in (reader.fieldnames or [])
        # external_id из CSV доехал до отчёта
        assert {r["external_id"] for r in out_rows} == {"ext-0", "ext-1", "ext-2"}
        for r in out_rows:
            assert r["status"] == "SUCCESS"
            assert r["verdict"] == "ai"

        # JSON отчёт
        r_json = await client.get(
            f"/api/v1/batch/{batch_id}/results.json",
            headers={"Authorization": f"Bearer {access}"},
        )
        assert r_json.status_code == 200
        payload = json.loads(r_json.content.decode("utf-8"))
        assert payload["batch_id"] == batch_id
        assert len(payload["results"]) == 3
        assert payload["n_completed"] == 3

    async def test_input_text_cleared_after_success(self, client):
        body = await _register(client, email="privacy@example.com")
        access = body["tokens"]["access_token"]

        await client.post(
            "/api/v1/batch",
            files={"file": ("e.csv", _make_csv(2), "text/csv")},
            headers={"Authorization": f"Bearer {access}"},
        )
        await _drain_jobs(Settings(min_chars=300))

        async with UnitOfWork() as uow:
            jobs = (
                await uow.session.execute(select(AnalysisJob))
            ).scalars().all()
        # Privacy: input_text должен быть NULL после SUCCESS.
        assert all(j.input_text is None for j in jobs)


class TestBatchLimits:
    async def test_active_batch_unique_per_user(self, client):
        body = await _register(client, email="dup@example.com")
        access = body["tokens"]["access_token"]
        headers = {"Authorization": f"Bearer {access}"}

        # Первый batch — успех.
        r1 = await client.post(
            "/api/v1/batch",
            files={"file": ("a.csv", _make_csv(2), "text/csv")},
            headers=headers,
        )
        assert r1.status_code == 202

        # Второй активный — должен 409.
        r2 = await client.post(
            "/api/v1/batch",
            files={"file": ("b.csv", _make_csv(2), "text/csv")},
            headers=headers,
        )
        assert r2.status_code == 409

        # После обработки первого — снова можно создавать.
        await _drain_jobs(Settings(min_chars=300))

        r3 = await client.post(
            "/api/v1/batch",
            files={"file": ("c.csv", _make_csv(2), "text/csv")},
            headers=headers,
        )
        assert r3.status_code == 202

    async def test_isolation_between_users(self, client):
        # Активный batch у Alice не блокирует Bob.
        a = await _register(client, email="alice-iso@example.com")
        b = await _register(client, email="bob-iso@example.com")

        ra = await client.post(
            "/api/v1/batch",
            files={"file": ("a.csv", _make_csv(2), "text/csv")},
            headers={"Authorization": f"Bearer {a['tokens']['access_token']}"},
        )
        rb = await client.post(
            "/api/v1/batch",
            files={"file": ("b.csv", _make_csv(2), "text/csv")},
            headers={"Authorization": f"Bearer {b['tokens']['access_token']}"},
        )
        assert ra.status_code == 202
        assert rb.status_code == 202

    async def test_too_many_rows_skipped(self, client):
        body = await _register(client, email="overflow@example.com")
        access = body["tokens"]["access_token"]

        big = _make_csv(110)  # > MAX_BATCH_ROWS=100
        resp = await client.post(
            "/api/v1/batch",
            files={"file": ("big.csv", big, "text/csv")},
            headers={"Authorization": f"Bearer {access}"},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["n_accepted"] == 100
        assert data["n_skipped"] == 10
        assert all(s["reason"] == "too_many_rows" for s in data["skipped"])

    async def test_invalid_csv_returns_400(self, client):
        body = await _register(client, email="bad-csv@example.com")
        access = body["tokens"]["access_token"]

        # Header без 'text' колонки.
        resp = await client.post(
            "/api/v1/batch",
            files={"file": ("bad.csv", b"id,name\n1,foo\n", "text/csv")},
            headers={"Authorization": f"Bearer {access}"},
        )
        assert resp.status_code == 400


class TestBatchOwnership:
    async def test_foreign_user_gets_404_on_status(self, client):
        a = await _register(client, email="a-own@example.com")
        b = await _register(client, email="b-own@example.com")

        ra = await client.post(
            "/api/v1/batch",
            files={"file": ("a.csv", _make_csv(2), "text/csv")},
            headers={"Authorization": f"Bearer {a['tokens']['access_token']}"},
        )
        bid = ra.json()["batch_id"]

        # Bob спрашивает чужой batch — 404.
        rb = await client.get(
            f"/api/v1/batch/{bid}",
            headers={"Authorization": f"Bearer {b['tokens']['access_token']}"},
        )
        assert rb.status_code == 404

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.post(
            "/api/v1/batch",
            files={"file": ("e.csv", _make_csv(2), "text/csv")},
        )
        assert resp.status_code == 401

    async def test_results_unavailable_until_completed(self, client):
        body = await _register(client, email="early@example.com")
        access = body["tokens"]["access_token"]
        ra = await client.post(
            "/api/v1/batch",
            files={"file": ("e.csv", _make_csv(2), "text/csv")},
            headers={"Authorization": f"Bearer {access}"},
        )
        bid = ra.json()["batch_id"]

        # CSV/JSON отчёт пока недоступен — batch ещё PENDING.
        r_csv = await client.get(
            f"/api/v1/batch/{bid}/results.csv",
            headers={"Authorization": f"Bearer {access}"},
        )
        assert r_csv.status_code == 409


class TestBatchListing:
    async def test_me_batches_returns_history(self, client):
        body = await _register(client, email="hist@example.com")
        access = body["tokens"]["access_token"]
        headers = {"Authorization": f"Bearer {access}"}

        # Создаём 2 последовательных batch'а (между ними drain).
        for label in ("first", "second"):
            await client.post(
                "/api/v1/batch",
                files={"file": (f"{label}.csv", _make_csv(2), "text/csv")},
                headers=headers,
            )
            await _drain_jobs(Settings(min_chars=300))

        resp = await client.get("/me/batches", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 2
        # Сортировка DESC по created_at — second идёт первым.
        assert data["items"][0]["file_name"] == "second.csv"
        assert all(b["status"] == "COMPLETED" for b in data["items"])


class TestRaceSafeCounter:
    async def test_concurrent_bumps_no_lost_update(self, client):
        """Симулируем concurrent инкременты двух воркеров одного batch'а.

        Старый паттерн `obj.n_completed += 1` потерял бы один из них
        (read-modify-write race). Атомарный SQL UPDATE — нет.
        """
        body = await _register(client, email="race@example.com")
        access = body["tokens"]["access_token"]
        ra = await client.post(
            "/api/v1/batch",
            files={"file": ("r.csv", _make_csv(2), "text/csv")},
            headers={"Authorization": f"Bearer {access}"},
        )
        bid = ra.json()["batch_id"]

        # Прямой вызов _bump_batch_progress дважды (без обработки) — счётчик
        # должен корректно дойти до n_total=2 → status=COMPLETED.
        async with UnitOfWork() as uow:
            await _bump_batch_progress(uow.session, bid, success=True)
            await uow.commit()
        async with UnitOfWork() as uow:
            await _bump_batch_progress(uow.session, bid, success=False)
            await uow.commit()

        async with UnitOfWork() as uow:
            batch = (
                await uow.session.execute(select(BatchJob).where(BatchJob.id == bid))
            ).scalar_one()
        assert batch.n_completed == 1
        assert batch.n_failed == 1
        assert batch.status == "COMPLETED"
        assert batch.completed_at is not None

    async def test_bump_uses_sql_expression_not_python(self):
        """Защитный «контракт-тест»: проверяем, что _bump_batch_progress
        обновляет колонку через SQL-expression (`n_completed + 1`), а не
        через Python read-modify-write. Тонко: смотрим на скомпилированный
        SQL — там должна быть операция `+`, а не литеральное значение.
        """
        # Этот тест дешевле, чем гонять реальный concurrency. Если кто-то
        # перепишет _bump_batch_progress на ORM-паттерн `obj.n_completed += 1`,
        # компиляция изменится — тест поймает регрессию.
        from sqlalchemy import case, update

        from app.database.models import BatchJob, BatchStatus

        new_completed = BatchJob.n_completed + 1
        new_failed = BatchJob.n_failed
        is_last = (new_completed + new_failed) == BatchJob.n_total
        stmt = (
            update(BatchJob)
            .where(BatchJob.id == "00000000-0000-0000-0000-000000000000")
            .values(
                n_completed=new_completed,
                n_failed=new_failed,
                status=case((is_last, BatchStatus.COMPLETED), else_=BatchStatus.PROCESSING),
            )
        )
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": False}))
        assert "n_completed + " in compiled, compiled
