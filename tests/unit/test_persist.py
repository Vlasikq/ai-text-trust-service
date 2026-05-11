"""Tests for database.persist — fallback log and response_to_model."""

import json
import threading
from unittest.mock import patch

from app.database import persist as persist_module
from app.database.models import Analysis
from app.database.persist import _write_fallback, persist_with_fallback, response_to_model
from app.schemas import (
    AnalyzeResponse,
    RiskLevel,
    Status,
    Verdict,
    WarningCode,
)


def _make_response(**overrides) -> AnalyzeResponse:
    defaults = {
        "request_id": "550e8400-e29b-41d4-a716-446655440000",
        "status": Status.success,
        "verdict": Verdict.ai,
        "confidence": 0.85,
        "risk_level": RiskLevel.high,
        "text_length": 500,
        "processing_time_ms": 42.0,
        "warnings": [WarningCode.text_truncated],
    }
    defaults.update(overrides)
    return AnalyzeResponse(**defaults)


class TestResponseToModel:
    def test_creates_analysis(self):
        resp = _make_response()
        model = response_to_model(
            response=resp,
            cleaned_text="Тестовый текст.",
            detector_method="tfidf",
            cascade_path="fast",
            inference_ms=12.0,
            cached=False,
            model_version="v1",
            service_version="0.1.0",
        )
        assert isinstance(model, Analysis)
        assert model.status == "SUCCESS"
        assert model.verdict == "ai"
        assert model.confidence == 0.85
        assert model.risk_level == "HIGH"
        assert model.detector_used == "tfidf"
        assert model.text_length == 500
        assert "TEXT_TRUNCATED" in model.warnings

    def test_no_decision(self):
        resp = _make_response(
            status=Status.no_decision,
            verdict=None,
            confidence=None,
            risk_level=None,
        )
        model = response_to_model(
            response=resp,
            cleaned_text="Короткий.",
            detector_method="tfidf",
            cascade_path=None,
            inference_ms=1.0,
            cached=False,
            model_version="v1",
            service_version="0.1.0",
        )
        assert model.status == "NO_DECISION"
        assert model.verdict is None

    def test_text_hash_computed(self):
        model = response_to_model(
            response=_make_response(),
            cleaned_text="Тестовый текст.",
            detector_method="tfidf",
            cascade_path=None,
            inference_ms=1.0,
            cached=False,
            model_version="v1",
            service_version="0.1.0",
        )
        assert len(model.text_hash) == 64  # SHA-256 hex


class TestFallbackLog:
    def test_writes_jsonl(self, tmp_path):
        model = response_to_model(
            response=_make_response(),
            cleaned_text="Тестовый текст.",
            detector_method="tfidf",
            cascade_path="fast",
            inference_ms=12.0,
            cached=False,
            model_version="v1",
            service_version="0.1.0",
        )

        log_path = tmp_path / "fallback.jsonl"
        _write_fallback(model, log_path)

        assert log_path.exists()
        line = log_path.read_text(encoding="utf-8").strip()
        data = json.loads(line)
        assert data["status"] == "SUCCESS"
        assert data["fallback_reason"] == "db_write_failed"

    def test_appends_multiple(self, tmp_path):
        log_path = tmp_path / "fallback.jsonl"
        for i in range(3):
            model = response_to_model(
                response=_make_response(),
                cleaned_text=f"Текст {i}.",
                detector_method="tfidf",
                cascade_path=None,
                inference_ms=1.0,
                cached=False,
                model_version="v1",
                service_version="0.1.0",
            )
            _write_fallback(model, log_path)

        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

    def test_creates_parent_dirs(self, tmp_path):
        log_path = tmp_path / "deep" / "nested" / "fallback.jsonl"
        model = response_to_model(
            response=_make_response(),
            cleaned_text="Текст.",
            detector_method="tfidf",
            cascade_path=None,
            inference_ms=1.0,
            cached=False,
            model_version="v1",
            service_version="0.1.0",
        )
        _write_fallback(model, log_path)
        assert log_path.exists()


# ── Integration: DB-write падает → переходим в fallback ──────


class _FailingUoW:
    """Мок UnitOfWork, который имитирует выпадение БД на commit().

    Поведение совпадает с реальным UoW: __aenter__/__aexit__ async,
    repos доступны, но commit() бросает.
    """

    def __init__(self, *_, **__):
        self.analyses = self  # save() ниже — заглушка

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    async def save(self, _analysis):
        return True

    async def commit(self):
        raise RuntimeError("simulated db outage")


class TestPersistWithFallback:
    async def test_db_failure_writes_to_fallback(self, tmp_path):
        """Главный graceful-degradation сценарий: БД упала во время commit() —
        analysis попадает в fallback-лог, исключение наружу не пробрасывается."""
        analysis = response_to_model(
            response=_make_response(),
            cleaned_text="Текст.",
            detector_method="tfidf",
            cascade_path="fast",
            inference_ms=12.0,
            cached=False,
            model_version="v1",
            service_version="0.1.0",
        )
        log_path = tmp_path / "db_fallback.jsonl"

        with patch("app.database.persist.UnitOfWork", _FailingUoW):
            await persist_with_fallback(analysis, log_path)

        assert log_path.exists()
        record = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert record["fallback_reason"] == "db_write_failed"
        assert record["request_id"] == str(analysis.request_id)
        assert record["status"] == "SUCCESS"

    async def test_db_success_no_fallback(self, tmp_path):
        """Если commit() прошёл — fallback-файл не создаётся."""

        class _OkUoW(_FailingUoW):
            async def commit(self):
                return None

        analysis = response_to_model(
            response=_make_response(),
            cleaned_text="Текст.",
            detector_method="tfidf",
            cascade_path=None,
            inference_ms=1.0,
            cached=False,
            model_version="v1",
            service_version="0.1.0",
        )
        log_path = tmp_path / "should_not_exist.jsonl"

        with patch("app.database.persist.UnitOfWork", _OkUoW):
            await persist_with_fallback(analysis, log_path)

        assert not log_path.exists()

    async def test_fallback_offloaded_to_thread(self, tmp_path):
        """_write_fallback вызывается через asyncio.to_thread — не из event-loop thread'а.

        Свойство «не блокирует loop» проверяем по факту переключения потока, а не
        по таймингам: иначе тест дребезжит на медленном CI.
        """
        main_thread = threading.current_thread()
        captured: dict = {}
        original_write = persist_module._write_fallback

        def capturing_write(analysis, path):
            captured["thread"] = threading.current_thread()
            original_write(analysis, path)

        analysis = response_to_model(
            response=_make_response(),
            cleaned_text="Текст.",
            detector_method="tfidf",
            cascade_path="fast",
            inference_ms=12.0,
            cached=False,
            model_version="v1",
            service_version="0.1.0",
        )
        log_path = tmp_path / "fallback.jsonl"

        with patch("app.database.persist.UnitOfWork", _FailingUoW), patch(
            "app.database.persist._write_fallback", capturing_write
        ):
            await persist_with_fallback(analysis, log_path)

        assert captured["thread"] is not main_thread, (
            "_write_fallback должен вызываться из worker-thread'а, не из event loop"
        )
        assert log_path.exists()
