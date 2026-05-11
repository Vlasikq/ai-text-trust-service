"""Tests for app.config — Settings defaults and overrides."""

import pytest

from app.config import DetectorType, Settings, get_settings


class TestSettingsDefaults:
    def test_default_detector_type(self):
        s = Settings()
        assert s.detector_type == DetectorType.tfidf

    def test_default_thresholds(self):
        s = Settings()
        assert s.risk_thresh_low == 0.30
        assert s.risk_thresh_high == 0.70

    def test_default_text_constraints(self):
        s = Settings()
        assert s.min_chars == 300
        assert s.max_chars == 8000

    def test_default_cascade(self):
        s = Settings()
        assert s.cascade_lo == 0.30
        assert s.cascade_hi == 0.70
        assert s.cascade_timeout_ms == 3000

    def test_default_calibration_enabled(self):
        s = Settings()
        assert s.calibration_enabled is True

    def test_default_log_text_false(self):
        s = Settings()
        assert s.log_text is False

    def test_default_cache_maxsize(self):
        s = Settings()
        assert s.cache_maxsize == 1024

    def test_artifacts_dir_is_path(self):
        s = Settings()
        assert s.artifacts_dir.name == "service"

    def test_transformer_dir_is_path(self):
        s = Settings()
        assert s.transformer_dir.name == "model"


class TestSettingsOverrides:
    def test_override_detector_type(self, monkeypatch):
        monkeypatch.setenv("DETECTOR_TYPE", "tfidf")
        s = Settings()
        assert s.detector_type == DetectorType.tfidf

    def test_override_min_chars(self, monkeypatch):
        monkeypatch.setenv("MIN_CHARS", "500")
        s = Settings()
        assert s.min_chars == 500

    def test_override_cascade_lo(self, monkeypatch):
        monkeypatch.setenv("CASCADE_LO", "0.20")
        s = Settings()
        assert s.cascade_lo == pytest.approx(0.20)

    def test_override_cascade_timeout_ms_via_request_timeout_alias(self, monkeypatch):
        monkeypatch.delenv("CASCADE_TIMEOUT_MS", raising=False)
        monkeypatch.setenv("REQUEST_TIMEOUT_MS", "4500")
        s = Settings()
        assert s.cascade_timeout_ms == 4500

    def test_override_cascade_timeout_ms_canonical_env(self, monkeypatch):
        monkeypatch.delenv("REQUEST_TIMEOUT_MS", raising=False)
        monkeypatch.setenv("CASCADE_TIMEOUT_MS", "6000")
        s = Settings()
        assert s.cascade_timeout_ms == 6000

    def test_override_log_level(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        s = Settings()
        assert s.log_level == "DEBUG"

    def test_extra_env_ignored(self, monkeypatch):
        monkeypatch.setenv("SOME_UNKNOWN_VAR", "whatever")
        # Чистим SERVICE_VERSION, чтобы default из Settings не перебивался .env-файлом.
        monkeypatch.delenv("SERVICE_VERSION", raising=False)
        s = Settings(_env_file=None)  # should not raise
        assert s.service_version == "0.3.0"


class TestDetectorTypeEnum:
    def test_all_values(self):
        assert set(DetectorType) == {
            DetectorType.tfidf,
            DetectorType.transformer,
            DetectorType.cascade,
        }

    def test_string_value(self):
        assert DetectorType.tfidf.value == "tfidf"


class TestGetSettings:
    def test_returns_settings_instance(self):
        s = get_settings()
        assert isinstance(s, Settings)
