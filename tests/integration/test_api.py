"""Tests for API endpoint POST /api/v1/analyze.

Uses FakeDetector из conftest — no real model artifacts needed.
/health и /ready покрыты в tests/test_health.py.
"""

import json

import pytest
from fastapi.testclient import TestClient

from tests.conftest import FakeDetector, FakeExplainer, make_test_app


@pytest.fixture
def client():
    return TestClient(make_test_app(detector=FakeDetector(prob_ai=0.85)))


@pytest.fixture
def low_client():
    return TestClient(make_test_app(detector=FakeDetector(prob_ai=0.10)))


# ── POST /api/v1/analyze ─────────────────────────────────────


class TestAnalyzeHappyPath:
    def test_success_response(self, client):
        text = "Длинный текст для проверки детектора."
        resp = client.post("/api/v1/analyze", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "SUCCESS"
        assert data["verdict"] == "ai"
        assert data["confidence"] == 0.85
        assert data["risk_level"] == "HIGH"

    def test_request_id_present(self, client):
        resp = client.post("/api/v1/analyze", json={"text": "Текст для анализа."})
        data = resp.json()
        assert len(data["request_id"]) == 36  # UUID4

    def test_disclaimer_present(self, client):
        resp = client.post("/api/v1/analyze", json={"text": "Текст для анализа."})
        data = resp.json()
        assert "вероятностный" in data["disclaimer"]

    def test_processing_time_positive(self, client):
        resp = client.post("/api/v1/analyze", json={"text": "Текст для анализа."})
        data = resp.json()
        assert data["processing_time_ms"] > 0

    def test_text_length_tracked(self, client):
        text = "Текст для анализа."
        resp = client.post("/api/v1/analyze", json={"text": text})
        data = resp.json()
        assert data["text_length"] == len(text)


class TestAnalyzeVerdicts:
    def test_human_verdict(self, low_client):
        resp = low_client.post("/api/v1/analyze", json={"text": "Текст для анализа."})
        data = resp.json()
        assert data["verdict"] == "human"
        assert data["risk_level"] == "LOW"

    def test_medium_risk(self):
        app = make_test_app(detector=FakeDetector(prob_ai=0.50))
        client = TestClient(app)
        resp = client.post("/api/v1/analyze", json={"text": "Текст для анализа."})
        data = resp.json()
        assert data["risk_level"] == "MEDIUM"
        assert "LOW_CONFIDENCE" in data["warnings"]


class TestAnalyzeShortText:
    def test_too_short_no_decision(self):
        app = make_test_app(min_chars=300)
        client = TestClient(app)
        resp = client.post("/api/v1/analyze", json={"text": "Короткий."})
        data = resp.json()
        assert data["status"] == "NO_DECISION"
        assert data["verdict"] is None
        assert "TEXT_TOO_SHORT" in data["warnings"]


class TestAnalyzeTruncation:
    def test_long_text_truncated(self):
        app = make_test_app(min_chars=10, max_chars=50)
        client = TestClient(app)
        long_text = "Первое предложение. Второе предложение. Третье предложение. Четвёртое."
        resp = client.post("/api/v1/analyze", json={"text": long_text})
        data = resp.json()
        assert data["was_truncated"] is True
        assert "TEXT_TRUNCATED" in data["warnings"]


class TestAnalyzeExplain:
    def test_explain_adds_heuristic_warning(self):
        app = make_test_app(
            detector=FakeDetector(prob_ai=0.85),
            min_chars=10,
            explainer=FakeExplainer(),
        )
        client = TestClient(app)
        resp = client.post(
            "/api/v1/analyze",
            json={"text": "Достаточно длинный текст для прохода min_chars.", "explain": True},
        )
        data = resp.json()
        assert data["explanation"] is not None
        assert "STYLE_EXPLANATION_HEURISTIC" in data["warnings"]


class TestAnalyzeDebug:
    def test_debug_includes_method_scores(self, client):
        resp = client.post("/api/v1/analyze", json={"text": "Текст для анализа.", "debug": True})
        data = resp.json()
        assert data["method_scores"] is not None
        assert len(data["method_scores"]) == 1
        assert data["method_scores"][0]["method"] == "tfidf"

    def test_no_debug_no_method_scores(self, client):
        resp = client.post("/api/v1/analyze", json={"text": "Текст для анализа."})
        data = resp.json()
        assert data["method_scores"] is None


class TestAnalyzeValidation:
    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param({"text": ""}, id="empty"),
            pytest.param({}, id="missing"),
            pytest.param({"text": "x" * 50_001}, id="too_long"),
        ],
    )
    def test_payload_rejected(self, client, payload):
        resp = client.post("/api/v1/analyze", json=payload)
        assert resp.status_code == 422


class TestAnalyzeErrorPaths:
    def test_detector_runtime_error_returns_500_without_traceback(self):
        """Безопасный 500: текст исключения и traceback не утекают клиенту."""

        class CrashingDetector(FakeDetector):
            def predict(self, text):
                raise RuntimeError("simulated detector crash with secret detail")

        # raise_server_exceptions=False: TestClient по умолчанию ре-райзит
        # серверные исключения, а нам нужен реальный response клиенту.
        client = TestClient(
            make_test_app(detector=CrashingDetector()),
            raise_server_exceptions=False,
        )
        resp = client.post(
            "/api/v1/analyze",
            json={"text": "Длинный текст для анализа."},
        )
        assert resp.status_code == 500
        body = resp.text
        assert "Traceback" not in body
        assert "RuntimeError" not in body
        assert "secret detail" not in body


# /health и /ready живут отдельно в tests/test_health.py — здесь не дублируем.


# ── Graceful degradation: persist при упавшей БД ──────────────


class _FailingUoW:
    """UoW, который имитирует выпадение БД на commit() — для теста persist-fallback."""

    def __init__(self, *_, **__):
        self.analyses = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    async def save(self, _analysis):
        return True

    async def commit(self):
        raise RuntimeError("simulated db outage")


class TestGracefulDbDown:
    """db_enabled=True + БД лежит → API возвращает 200, persist уходит в fallback-лог."""

    def test_analyze_succeeds_with_db_down_and_writes_fallback(self, tmp_path, monkeypatch):
        fallback_path = tmp_path / "db_fallback.jsonl"
        app = make_test_app(
            detector=FakeDetector(prob_ai=0.85),
            db_fallback_log_path=fallback_path,
        )
        # Включаем DB-ветку persist в analyze.py, не поднимая реальный engine.
        app.state.db_enabled = True
        app.state.deployment_id = None
        monkeypatch.setattr("app.database.persist.UnitOfWork", _FailingUoW)

        client = TestClient(app)
        resp = client.post(
            "/api/v1/analyze",
            json={"text": "Достаточно длинный текст для анализа."},
        )

        assert resp.status_code == 200
        # TestClient выполняет BackgroundTasks синхронно после возврата response.
        assert fallback_path.exists(), "persist-fallback не сработал"
        record = json.loads(fallback_path.read_text(encoding="utf-8").strip())
        assert record["fallback_reason"] == "db_write_failed"
        assert record["request_id"] == resp.json()["request_id"]
