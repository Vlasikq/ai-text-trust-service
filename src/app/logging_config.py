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


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with JSON formatter and correlation_id filter.

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

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
