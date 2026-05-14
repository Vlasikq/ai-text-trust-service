"""Структурированное JSON-логирование с прокидыванием `correlation_id` через ContextVar.

В каждую запись лога подмешивается сквозной `correlation_id` запроса. ID
приходит из заголовка `X-Correlation-ID` или `X-Request-ID`, либо
генерируется в `RequestLoggingMiddleware`. Хендлеры, фоновые задачи и
репозитории пишут логи стандартным `logging`, а фильтр подставляет id
из ContextVar автоматически.
"""

import logging
from contextvars import ContextVar

from pythonjsonlogger.json import JsonFormatter

# ContextVar для correlation_id.

correlation_id_var: ContextVar[str | None] = ContextVar(
    "aitrust_correlation_id", default=None
)


def get_correlation_id() -> str | None:
    return correlation_id_var.get()


def set_correlation_id(value: str | None) -> None:
    correlation_id_var.set(value)


# Фильтр и настройка логирования.


class CorrelationIdFilter(logging.Filter):
    """Подмешивает `correlation_id` из ContextVar в атрибуты log-записи."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id_var.get() or "-"
        return True


class SensitiveDataFilter(logging.Filter):
    """Заменяет значения чувствительных полей в `extra`/`args` на `***REDACTED***`.

    Ловит структурный лог `log.info("login", extra={"password": pw})`.
    Не ловит сборку через f-string или `%s` в самом сообщении — это решается
    тем, что мы пишем структурный JSON-лог везде в проекте.
    """

    SENSITIVE_FIELDS = frozenset(
        {
            "password",
            "password_hash",
            "jwt_secret",
            "auth_jwt_secret",
            "secret",
            "access_token",
            "refresh_token",
            "token",
            "authorization",
            "bearer",
            "pg_password",
            "database_url",  # содержит пароль БД в DSN
        }
    )
    REDACTED = "***REDACTED***"

    def _scrub(self, value: object) -> object:
        if isinstance(value, dict):
            return {
                k: (self.REDACTED if k.lower() in self.SENSITIVE_FIELDS else self._scrub(v))
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            scrubbed = [self._scrub(v) for v in value]
            return type(value)(scrubbed)
        return value

    def filter(self, record: logging.LogRecord) -> bool:
        # `extra={...}` оседает прямо в record.__dict__ как атрибуты.
        for key in list(record.__dict__.keys()):
            if key.lower() in self.SENSITIVE_FIELDS:
                record.__dict__[key] = self.REDACTED
            else:
                val = record.__dict__[key]
                if isinstance(val, dict):
                    record.__dict__[key] = self._scrub(val)
        if isinstance(record.args, dict):
            record.args = self._scrub(record.args)
        return True


def setup_logging(level: str = "INFO") -> None:
    """Настраивает root-логгер: JSON-форматтер плюс фильтр для correlation_id.

    Идемпотентно: очищает существующие handler'ы — безопасно вызывать повторно
    (например, в тестах при множественных стартах приложения).
    """
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter(
            "{asctime}{levelname}{name}{message}{correlation_id}",
            style="{",
            rename_fields={
                "asctime": "timestamp",
                "levelname": "level",
                "name": "logger",
            },
            # ensure_ascii=False → кириллица в логах остаётся читаемой и в pwsh,
            # и в Cloud Logging YC (UTF-8 surrogates агрегаторы понимают хуже).
            json_ensure_ascii=False,
        )
    )
    handler.addFilter(CorrelationIdFilter())
    handler.addFilter(SensitiveDataFilter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
