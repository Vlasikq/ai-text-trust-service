"""Тесты middleware, metrics, cache в API и сценариев деградации."""

import io
import json
import logging
import time

from fastapi.testclient import TestClient

from app.detectors.base import BaseDetector, DetectionResult
from app.logging_config import setup_logging
from tests.conftest import FakeDetector, make_test_app


def _app(**kw):
    """Шорткат: analyze+health роутеры с дефолтным FakeDetector."""
    kw.setdefault("routers", ("analyze", "health"))
    return make_test_app(**kw)


# ── Cache integration ─────────────────────────────────────────


class TestCacheIntegration:
    def test_second_request_uses_cache(self):
        app = _app()
        client = TestClient(app)
        text = "Один и тот же текст для кэша."
        resp1 = client.post("/api/v1/analyze", json={"text": text})
        resp2 = client.post("/api/v1/analyze", json={"text": text})
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        # Cache stats теперь публикуются через /ready
        ready = client.get("/ready").json()
        assert ready["components"]["cache"]["hits"] >= 1

    def test_different_texts_no_cache_hit(self):
        app = _app()
        client = TestClient(app)
        client.post("/api/v1/analyze", json={"text": "Текст первый."})
        client.post("/api/v1/analyze", json={"text": "Текст второй."})
        ready = client.get("/ready").json()
        assert ready["components"]["cache"]["hits"] == 0
        assert ready["components"]["cache"]["misses"] == 2


# ── Degradation scenarios ─────────────────────────────────────


class TestDegradationScenarios:
    def test_short_text_no_decision(self):
        app = _app(min_chars=300)
        client = TestClient(app)
        resp = client.post("/api/v1/analyze", json={"text": "Короткий."})
        data = resp.json()
        assert data["status"] == "NO_DECISION"
        assert "TEXT_TOO_SHORT" in data["warnings"]

    def test_truncation_warning(self):
        app = _app(min_chars=10, max_chars=50)
        client = TestClient(app)
        long = "Первое предложение. Второе предложение. Третье предложение. Четвёртое."
        data = client.post("/api/v1/analyze", json={"text": long}).json()
        assert data["was_truncated"] is True
        assert "TEXT_TRUNCATED" in data["warnings"]

    def test_transformer_unavailable_warning(self):
        detector = FakeDetector(
            prob_ai=0.5,
            metadata={"cascade_path": "fallback", "warning": "TRANSFORMER_UNAVAILABLE"},
        )
        app = _app(detector=detector)
        client = TestClient(app)
        data = client.post("/api/v1/analyze", json={"text": "Текст для анализа."}).json()
        assert "TRANSFORMER_UNAVAILABLE" in data["warnings"]

    def test_low_confidence_warning(self):
        detector = FakeDetector(prob_ai=0.50)
        app = _app(detector=detector)
        client = TestClient(app)
        data = client.post("/api/v1/analyze", json={"text": "Текст для анализа."}).json()
        assert data["risk_level"] == "MEDIUM"
        assert "LOW_CONFIDENCE" in data["warnings"]

    def test_ready_returns_503_when_detector_not_ready(self):
        app = _app(detector=FakeDetector(ready=False))
        client = TestClient(app)
        resp = client.get("/ready")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "not_ready"
        assert body["components"]["detector"]["ready"] is False
        # Liveness при этом обязан оставаться зелёным.
        assert client.get("/health").status_code == 200

    def test_inference_timeout(self):
        """Detector that sleeps longer than cascade_timeout_ms → ERROR."""

        class SlowDetector(BaseDetector):
            def load(self):
                pass

            def predict(self, text):
                # 5 секунд — запас над фактическим таймаутом 3с из dev .env
                # (kwarg cascade_timeout_ms=100 перебивается pydantic-settings).
                # Меньшее значение начинает флакать.
                time.sleep(5)
                return DetectionResult(prob_ai=0.5, method="slow", inference_ms=5000)

            def is_ready(self):
                return True

        app = _app(detector=SlowDetector(), cascade_timeout_ms=100)
        client = TestClient(app)
        data = client.post("/api/v1/analyze", json={"text": "Текст для анализа."}).json()
        assert data["status"] == "ERROR"
        assert "INFERENCE_TIMEOUT" in data["warnings"]


# ── Correlation ID propagation ────────────────────────────────


class TestCorrelationId:
    def test_generated_when_no_header(self):
        client = TestClient(_app(with_middleware=True))
        resp = client.post("/api/v1/analyze", json={"text": "Текст для анализа."})
        cid = resp.headers["X-Correlation-ID"]
        assert len(cid) == 36  # UUID4
        # request_id в response совпадает с заголовком
        assert resp.json()["request_id"] == cid

    def test_uses_client_correlation_id(self):
        client = TestClient(_app(with_middleware=True))
        provided = "11111111-1111-4111-8111-111111111111"
        resp = client.post(
            "/api/v1/analyze",
            json={"text": "Текст для анализа."},
            headers={"X-Correlation-ID": provided},
        )
        assert resp.headers["X-Correlation-ID"] == provided
        assert resp.json()["request_id"] == provided

    def test_falls_back_to_x_request_id(self):
        client = TestClient(_app(with_middleware=True))
        provided = "22222222-2222-4222-8222-222222222222"
        resp = client.post(
            "/api/v1/analyze",
            json={"text": "Текст для анализа."},
            headers={"X-Request-ID": provided},
        )
        assert resp.headers["X-Correlation-ID"] == provided
        assert resp.json()["request_id"] == provided

    def test_unique_per_request_when_no_header(self):
        client = TestClient(_app(with_middleware=True))
        r1 = client.post("/api/v1/analyze", json={"text": "Первый текст."})
        r2 = client.post("/api/v1/analyze", json={"text": "Второй текст."})
        assert r1.headers["X-Correlation-ID"] != r2.headers["X-Correlation-ID"]

    def test_id_returned_for_health(self):
        """Liveness/readiness тоже получают correlation_id (для трейсинга проб)."""
        client = TestClient(_app(with_middleware=True))
        resp = client.get("/health")
        assert "X-Correlation-ID" in resp.headers


# ── JSON-формат логов ─────────────────────────────────────────


class TestJsonLogging:
    """setup_logging() ставит JsonFormatter и инжектит correlation_id в каждую запись."""

    def _capture_log(self, fn) -> list[dict]:
        """Привязать StreamHandler к буферу, вызвать fn, разобрать NDJSON."""
        # setup_logging уже подменил root handler; перевешиваем его на буфер.
        setup_logging("INFO")
        root = logging.getLogger()
        original = root.handlers[0]
        buffer = io.StringIO()
        new_handler = logging.StreamHandler(buffer)
        new_handler.setFormatter(original.formatter)
        for f in original.filters:
            new_handler.addFilter(f)
        root.handlers = [new_handler]
        try:
            fn()
        finally:
            new_handler.flush()
        return [
            json.loads(line) for line in buffer.getvalue().splitlines() if line.strip()
        ]

    def test_log_record_is_valid_json_with_correlation_id(self):
        client = TestClient(_app(with_middleware=True))
        provided = "33333333-3333-4333-8333-333333333333"

        records = self._capture_log(
            lambda: client.post(
                "/api/v1/analyze",
                json={"text": "Текст для анализа."},
                headers={"X-Correlation-ID": provided},
            )
        )

        # Найдём строку aitrust.metrics — её точно пишет middleware.
        request_logs = [
            r for r in records if r.get("logger") == "aitrust.metrics"
        ]
        assert request_logs, "middleware не записал access-лог"
        assert all(r.get("correlation_id") == provided for r in request_logs)
        assert all("level" in r and "timestamp" in r for r in request_logs)

    def test_log_outside_request_has_dash_correlation_id(self):
        """Логи без активного запроса получают placeholder '-' вместо id."""
        records = self._capture_log(lambda: logging.getLogger("aitrust").info("orphan"))
        assert records, "ничего не залогировалось"
        assert records[0]["correlation_id"] == "-"


# ── Rate-limit smoke ──────────────────────────────────────────


class TestRateLimitDecorators:
    """Включаем limiter временно и проверяем что декораторы на /analyze и /jobs
    реально срабатывают — без этого регрессия в callable-провайдере лимита
    (slowapi вызывает provider() без аргументов) проходит мимо unit-тестов.
    """

    def _enable_limiter(self, monkeypatch):
        from app.middleware.ratelimit import limiter

        # Чистим storage чтобы тесты не зависели от других прогонов.
        try:
            limiter._storage.reset()  # type: ignore[attr-defined]
        except Exception:
            pass
        monkeypatch.setattr(limiter, "enabled", True)

    def test_analyze_rate_limit_decorator_invokes_without_error(self, monkeypatch):
        # Поднимаем лимит до большого значения, чтобы тест не упёрся в 429:
        # цель — поймать регрессию вида «slowapi-провайдер вызван неправильно».
        monkeypatch.setenv("RATE_LIMIT_ANALYZE", "1000/minute")
        from app.config import get_settings

        get_settings.cache_clear()
        self._enable_limiter(monkeypatch)

        client = TestClient(_app())
        # Anon-путь.
        r1 = client.post("/api/v1/analyze", json={"text": "Текст для анализа."})
        assert r1.status_code == 200, r1.text
        # Auth-путь (фейк-токен — key_func разделит в auth-bucket по префиксу).
        r2 = client.post(
            "/api/v1/analyze",
            json={"text": "Текст для анализа."},
            headers={"Authorization": "Bearer faketoken"},
        )
        assert r2.status_code == 200, r2.text

    def test_analyze_rate_limit_returns_429_on_burst(self, monkeypatch):
        monkeypatch.setenv("RATE_LIMIT_ANALYZE", "2/minute")
        from app.config import get_settings

        get_settings.cache_clear()
        self._enable_limiter(monkeypatch)

        from slowapi.errors import RateLimitExceeded

        from app.middleware.ratelimit import rate_limit_exceeded_handler

        app = _app()
        app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
        client = TestClient(app)

        codes = [
            client.post("/api/v1/analyze", json={"text": f"Текст {i}."}).status_code
            for i in range(4)
        ]
        # Хотя бы один 429 должен прилететь — bucket по IP исчерпан после 2 запросов.
        assert 429 in codes, f"ожидали 429 в серии, получили {codes}"

    def test_429_response_has_retry_after_header(self, monkeypatch):
        monkeypatch.setenv("RATE_LIMIT_ANALYZE", "1/minute")
        from app.config import get_settings

        get_settings.cache_clear()
        self._enable_limiter(monkeypatch)

        from slowapi.errors import RateLimitExceeded

        from app.middleware.ratelimit import rate_limit_exceeded_handler

        app = _app()
        app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
        client = TestClient(app)

        client.post("/api/v1/analyze", json={"text": "Первый."})
        second = client.post("/api/v1/analyze", json={"text": "Второй."})
        assert second.status_code == 429
        assert "retry-after" in {k.lower() for k in second.headers.keys()}

    def test_anon_and_auth_buckets_independent(self, monkeypatch):
        # Bucket делится по префиксу ключа: anon → ip:..., auth → auth:<token[:32]>.
        # Один IP не должен делить лимит с авторизованным запросом.
        monkeypatch.setenv("RATE_LIMIT_ANALYZE", "1/minute")
        from app.config import get_settings

        get_settings.cache_clear()
        self._enable_limiter(monkeypatch)

        from slowapi.errors import RateLimitExceeded

        from app.middleware.ratelimit import rate_limit_exceeded_handler

        app = _app()
        app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
        client = TestClient(app)

        anon = client.post("/api/v1/analyze", json={"text": "Anon."})
        assert anon.status_code == 200
        anon_burst = client.post("/api/v1/analyze", json={"text": "Anon-2."})
        assert anon_burst.status_code == 429
        # Авторизованный запрос с тем же IP — другой bucket, должен пройти.
        auth = client.post(
            "/api/v1/analyze",
            json={"text": "Auth."},
            headers={"Authorization": "Bearer differentbucket"},
        )
        assert auth.status_code == 200
