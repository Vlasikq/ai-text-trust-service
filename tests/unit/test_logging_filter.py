"""Тесты SensitiveDataFilter: чувствительные поля в логах редактируются."""

import logging

import pytest

from app.logging_config import SensitiveDataFilter


@pytest.fixture
def filt() -> SensitiveDataFilter:
    return SensitiveDataFilter()


def _record(msg: str, **extras) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg=msg, args=None, exc_info=None,
    )
    for k, v in extras.items():
        setattr(record, k, v)
    return record


class TestSensitiveDataFilter:
    def test_password_in_extra_redacted(self, filt):
        rec = _record("login", password="hunter2")
        filt.filter(rec)
        assert rec.password == "***REDACTED***"

    def test_refresh_token_redacted(self, filt):
        rec = _record("token issued", refresh_token="abc.def.ghi")
        filt.filter(rec)
        assert rec.refresh_token == "***REDACTED***"

    def test_access_token_redacted(self, filt):
        rec = _record("auth", access_token="eyJhbGc...")
        filt.filter(rec)
        assert rec.access_token == "***REDACTED***"

    def test_jwt_secret_redacted(self, filt):
        rec = _record("config dump", jwt_secret="64-byte-hex-secret")
        filt.filter(rec)
        assert rec.jwt_secret == "***REDACTED***"

    def test_authorization_header_redacted(self, filt):
        rec = _record("request", authorization="Bearer eyJ...")
        filt.filter(rec)
        assert rec.authorization == "***REDACTED***"

    def test_database_url_redacted(self, filt):
        rec = _record("init", database_url="postgresql+asyncpg://u:p@host/db")
        filt.filter(rec)
        assert rec.database_url == "***REDACTED***"

    def test_case_insensitive_match(self, filt):
        rec = _record("x", Password="abc", REFRESH_TOKEN="def")
        filt.filter(rec)
        assert rec.Password == "***REDACTED***"
        assert rec.REFRESH_TOKEN == "***REDACTED***"

    def test_normal_fields_preserved(self, filt):
        rec = _record("ok", email="u@example.com", user_id="123", verdict="ai")
        filt.filter(rec)
        assert rec.email == "u@example.com"
        assert rec.user_id == "123"
        assert rec.verdict == "ai"

    def test_nested_dict_scrubbed(self, filt):
        rec = _record("event", payload={"email": "u@x.com", "password": "secret"})
        filt.filter(rec)
        assert rec.payload == {"email": "u@x.com", "password": "***REDACTED***"}

    def test_args_dict_scrubbed(self, filt):
        rec = _record("user %(email)s pw=%(password)s")
        rec.args = {"email": "u@x.com", "password": "secret"}
        filt.filter(rec)
        assert rec.args == {"email": "u@x.com", "password": "***REDACTED***"}
