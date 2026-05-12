"""Тесты POST /api/v1/jobs и GET /api/v1/jobs/{id}.

UnitOfWork здесь мокается через unittest.mock — осознанный паттерн: реальная
БД-логика покрывается тестами JobRepository (test_jobs_repository.py), а здесь
мы фокусируемся на роутер-слое (валидация, статус-коды, формирование poll_url,
проброс privacy-полей наружу). Real-DB end-to-end проходит через test_batch_api.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from uuid_utils import uuid7

from app.auth.dependencies import get_current_user_optional
from app.database.models import AnalysisJob, JobStatus, User, UserRole, UserStatus
from tests.conftest import make_test_app


def _fake_job(status: str = JobStatus.PENDING, **overrides) -> AnalysisJob:
    """Build AnalysisJob, готовый для возврата из мокнутого репозитория."""
    job = AnalysisJob(
        id=uuid7(),
        request_id=uuid7(),
        input_text="секретный текст пользователя",
        text_hash="b" * 64,
        text_length=200,
        detector_type="cascade",
        explain=False,
        status=status,
        created_at=datetime.now(timezone.utc),
    )
    for k, v in overrides.items():
        setattr(job, k, v)
    return job


@pytest.fixture
def mock_uow():
    """Подменяем UnitOfWork в jobs router."""
    jobs_repo = AsyncMock()
    analyses_repo = AsyncMock()

    uow = AsyncMock()
    uow.jobs = jobs_repo
    uow.analyses = analyses_repo
    uow.commit = AsyncMock()
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=False)

    with patch("app.routers.jobs.UnitOfWork", return_value=uow):
        yield uow


# ── POST /api/v1/jobs ────────────────────────────────────────


class TestCreateJob:
    def test_creates_pending_job_and_returns_202(self, mock_uow):
        new_job = _fake_job(status=JobStatus.PENDING)
        mock_uow.jobs.create = AsyncMock(return_value=new_job)

        client = TestClient(make_test_app(routers=("jobs",),db_enabled=True))
        resp = client.post("/api/v1/jobs", json={"text": "Достаточно длинный текст для анализа."})

        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "PENDING"
        assert data["job_id"] == str(new_job.id)
        assert data["poll_url"] == f"/api/v1/jobs/{new_job.id}"
        assert "request_id" in data

    def test_503_when_db_disabled(self):
        client = TestClient(make_test_app(routers=("jobs",),db_enabled=False))
        resp = client.post("/api/v1/jobs", json={"text": "Текст для анализа."})
        assert resp.status_code == 503
        assert "Database not available" in resp.json()["detail"]

    def test_idempotent_via_correlation_id(self, mock_uow):
        """С тем же X-Correlation-ID dva раза — возвращается тот же job (через UNIQUE-FK).

        В реальности это проверяется на уровне JobRepository.create_idempotent_on_duplicate
        (см. test_jobs_repository); здесь — что endpoint просто пробрасывает request_id."""
        existing = _fake_job(status=JobStatus.PENDING)
        mock_uow.jobs.create = AsyncMock(return_value=existing)

        client = TestClient(make_test_app(routers=("jobs",),db_enabled=True))
        cid = "11111111-1111-4111-8111-111111111111"
        resp = client.post(
            "/api/v1/jobs",
            json={"text": "Достаточно длинный текст для анализа."},
            headers={"X-Correlation-ID": cid},
        )

        assert resp.status_code == 202
        # request_id в response — то что вернул repository (UNIQUE)
        assert resp.json()["request_id"] == str(existing.request_id)

    def test_empty_text_rejected(self):
        client = TestClient(make_test_app(routers=("jobs",),db_enabled=True))
        resp = client.post("/api/v1/jobs", json={"text": ""})
        assert resp.status_code == 422

    def test_too_long_text_rejected(self):
        client = TestClient(make_test_app(routers=("jobs",),db_enabled=True))
        resp = client.post("/api/v1/jobs", json={"text": "x" * 50_001})
        assert resp.status_code == 422


# ── GET /api/v1/jobs/{id} ────────────────────────────────────


class TestGetJob:
    def test_returns_pending_status(self, mock_uow):
        job = _fake_job(status=JobStatus.PENDING)
        mock_uow.jobs.get_by_id = AsyncMock(return_value=job)

        client = TestClient(make_test_app(routers=("jobs",),db_enabled=True))
        resp = client.get(f"/api/v1/jobs/{job.id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "PENDING"
        assert data["job_id"] == str(job.id)
        assert data["result"] is None
        assert data["error"] is None

    def test_returns_404_when_not_found(self, mock_uow):
        mock_uow.jobs.get_by_id = AsyncMock(return_value=None)

        client = TestClient(make_test_app(routers=("jobs",),db_enabled=True))
        resp = client.get(f"/api/v1/jobs/{uuid7()}")
        assert resp.status_code == 404

    def test_503_when_db_disabled(self):
        client = TestClient(make_test_app(routers=("jobs",),db_enabled=False))
        resp = client.get(f"/api/v1/jobs/{uuid7()}")
        assert resp.status_code == 503

    def test_returns_error_message_when_failed(self, mock_uow):
        job = _fake_job(
            status=JobStatus.ERROR,
            error_message="Inference timeout after 30000 ms",
            finished_at=datetime.now(timezone.utc),
        )
        mock_uow.jobs.get_by_id = AsyncMock(return_value=job)

        client = TestClient(make_test_app(routers=("jobs",),db_enabled=True))
        resp = client.get(f"/api/v1/jobs/{job.id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ERROR"
        assert "timeout" in data["error"]
        assert data["result"] is None

    # ── Ownership / BOLA ───────────────────────────────────

    def _make_user(self, **overrides) -> User:
        return User(
            id=overrides.pop("id", uuid.uuid4()),
            email=overrides.pop("email", "u@example.com"),
            password_hash="$argon2id$dummy",
            status=UserStatus.ACTIVE,
            role=UserRole.USER,
            **overrides,
        )

    def test_owner_can_read_authored_job(self, mock_uow):
        owner = self._make_user()
        job = _fake_job(status=JobStatus.PENDING, user_id=str(owner.id))
        mock_uow.jobs.get_by_id = AsyncMock(return_value=job)

        app = make_test_app(routers=("jobs",), db_enabled=True)
        app.dependency_overrides[get_current_user_optional] = lambda: owner
        try:
            resp = TestClient(app).get(f"/api/v1/jobs/{job.id}")
        finally:
            app.dependency_overrides.pop(get_current_user_optional, None)
        assert resp.status_code == 200
        assert resp.json()["job_id"] == str(job.id)

    def test_other_user_gets_404_on_authored_job(self, mock_uow):
        # Чужой авторский job не должен утечь даже залогиненному пользователю.
        # 404, а не 403 — иначе утекает факт существования id.
        owner = self._make_user()
        intruder = self._make_user(email="x@example.com")
        job = _fake_job(status=JobStatus.SUCCESS, user_id=str(owner.id))
        mock_uow.jobs.get_by_id = AsyncMock(return_value=job)

        app = make_test_app(routers=("jobs",), db_enabled=True)
        app.dependency_overrides[get_current_user_optional] = lambda: intruder
        try:
            resp = TestClient(app).get(f"/api/v1/jobs/{job.id}")
        finally:
            app.dependency_overrides.pop(get_current_user_optional, None)
        assert resp.status_code == 404

    def test_anonymous_request_gets_404_on_authored_job(self, mock_uow):
        job = _fake_job(status=JobStatus.SUCCESS, user_id=str(uuid.uuid4()))
        mock_uow.jobs.get_by_id = AsyncMock(return_value=job)

        client = TestClient(make_test_app(routers=("jobs",), db_enabled=True))
        resp = client.get(f"/api/v1/jobs/{job.id}")
        assert resp.status_code == 404

    def test_anonymous_job_visible_by_uuid(self, mock_uow):
        # Анонимный клиент знает только UUID — без авторизации он должен
        # забрать свой же результат.
        job = _fake_job(status=JobStatus.PENDING, user_id=None)
        mock_uow.jobs.get_by_id = AsyncMock(return_value=job)

        client = TestClient(make_test_app(routers=("jobs",), db_enabled=True))
        resp = client.get(f"/api/v1/jobs/{job.id}")
        assert resp.status_code == 200

    def test_returns_result_when_success(self, mock_uow):
        """Когда status=SUCCESS, endpoint подгружает Analysis и возвращает AnalyzeResponse."""
        from app.database.models import Analysis

        analysis = Analysis(
            id=uuid7(),
            request_id=uuid7(),
            text_hash="c" * 64,
            text_length=300,
            status="SUCCESS",
            verdict="ai",
            confidence=0.85,
            risk_level="HIGH",
            detector_used="cascade",
            cascade_path="slow",
            inference_ms=8000.0,
            cached=False,
            model_version="ru-detector-0.1.0",
            service_version="0.1.0",
            warnings=[],
            requested_at=datetime.now(timezone.utc),
        )
        job = _fake_job(
            status=JobStatus.SUCCESS,
            request_id=analysis.request_id,
            analysis_id=analysis.id,
            finished_at=datetime.now(timezone.utc),
            input_text=None,  # очищено privacy-фильтром
        )
        mock_uow.jobs.get_by_id = AsyncMock(return_value=job)
        mock_uow.analyses.get_by_request_id = AsyncMock(return_value=analysis)

        client = TestClient(make_test_app(routers=("jobs",),db_enabled=True))
        resp = client.get(f"/api/v1/jobs/{job.id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "SUCCESS"
        assert data["result"] is not None
        assert data["result"]["verdict"] == "ai"
        assert data["result"]["confidence"] == 0.85
        assert data["result"]["risk_level"] == "HIGH"
