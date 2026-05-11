"""Tests for liveness (/health) and readiness (/ready) probes.

`/health` — всегда 200 если процесс отвечает, минимум полей.
`/ready`  — 200 когда detector.is_ready() и (если db_enabled) БД пингуется; иначе 503.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from tests.conftest import FakeDetector, make_test_app


def _app(detector=None, db_enabled=False, **state):
    """Локальный шорткат: health-роутер + дефолтный готовый FakeDetector."""
    return make_test_app(
        routers=("health",),
        detector=detector if detector is not None else FakeDetector(),
        db_enabled=db_enabled,
        calibrator=state.get("calibrator"),
        explainer=state.get("explainer"),
        cache_maxsize=128,
    )


# ── /health (liveness) ────────────────────────────────────────


class TestLiveness:
    def test_returns_200_when_detector_ready(self):
        client = TestClient(_app())
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "service_version" in body

    def test_returns_200_even_when_detector_not_ready(self):
        """Liveness не должен зависеть от состояния детектора —
        процесс жив, значит endpoint обязан ответить 200."""
        client = TestClient(_app(detector=FakeDetector(ready=False)))
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_minimal_payload(self):
        """Liveness — минимум полей; не должен лезть в детали компонентов."""
        client = TestClient(_app())
        body = client.get("/health").json()
        assert set(body.keys()) == {"status", "service_version"}


# ── /ready (readiness) ────────────────────────────────────────


class TestReadiness:
    def test_ready_when_detector_ready_and_db_disabled(self):
        client = TestClient(_app(db_enabled=False))
        resp = client.get("/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready"
        assert body["components"]["detector"]["ready"] is True
        assert body["components"]["db"]["enabled"] is False
        assert body["components"]["db"]["ready"] is None

    def test_503_when_detector_not_ready(self):
        client = TestClient(_app(detector=FakeDetector(ready=False), db_enabled=False))
        resp = client.get("/ready")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "not_ready"
        assert body["components"]["detector"]["ready"] is False

    def test_503_when_db_enabled_but_unreachable(self):
        client = TestClient(_app(db_enabled=True))
        # _ping_db поймает RuntimeError из get_engine() — engine не инициализирован
        resp = client.get("/ready")
        assert resp.status_code == 503
        body = resp.json()
        assert body["components"]["db"]["enabled"] is True
        assert body["components"]["db"]["ready"] is False

    def test_ready_when_db_enabled_and_reachable(self):
        async def fake_ping():
            return True

        with patch("app.routers.health._ping_db", fake_ping):
            client = TestClient(_app(db_enabled=True))
            resp = client.get("/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready"
        assert body["components"]["db"]["ready"] is True

    def test_components_reflect_optional_state(self):
        """calibrator/explainer показывают свой статус, но не блокируют readiness."""
        client = TestClient(_app(db_enabled=False))  # без calibrator/explainer
        body = client.get("/ready").json()
        assert body["components"]["calibrator"]["ready"] is False
        assert body["components"]["explainer"]["ready"] is False
        assert body["status"] == "ready"  # они опциональные

    def test_components_include_cache_stats(self):
        client = TestClient(_app(db_enabled=False))
        body = client.get("/ready").json()
        cache = body["components"]["cache"]
        assert cache is not None
        assert "hits" in cache and "misses" in cache

    def test_detector_info_included(self):
        client = TestClient(_app(db_enabled=False))
        body = client.get("/ready").json()
        info = body["components"]["detector"]["info"]
        assert info is not None
        assert info["ready"] is True


# ── _ping_db unit ────────────────────────────────────────────


class TestPingDb:
    async def test_returns_false_when_engine_uninitialized(self, monkeypatch):
        """Когда engine не инициализирован — _ping_db ловит исключение, возвращает False.

        Mокируем engine._engine/async_session через monkeypatch (вместо прямого
        присвоения приватных модульных переменных) — pytest откатит автоматически
        при teardown, не загрязняя других тестов.
        """
        from app.database import engine as engine_mod
        from app.routers.health import _ping_db

        monkeypatch.setattr(engine_mod, "_engine", None)
        monkeypatch.setattr(engine_mod, "async_session", None)

        assert await _ping_db() is False
