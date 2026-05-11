"""Метрики Prometheus и структурированное JSON-логирование запросов.

Поднимается четыре серии (см. `service-roadmap.md`):

  - `aitrust_requests_total` — counter с метками `status` и `risk_level`;
  - `aitrust_latency_seconds` — histogram латентности;
  - `aitrust_confidence` — histogram уверенности;
  - `aitrust_cascade_path` — counter путей каскада.

Текст пользователя в логи не попадает никогда.
"""

import logging
import time
import uuid

from fastapi import Request, Response
from prometheus_client import Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

from app.logging_config import set_correlation_id

CORRELATION_ID_HEADER = "X-Correlation-ID"
# Альтернативный заголовок — ALB / Traefik часто используют X-Request-ID
ALT_CORRELATION_HEADERS = ("X-Request-ID",)

log = logging.getLogger("aitrust.metrics")

# Метрики Prometheus.

REQUESTS_TOTAL = Counter(
    "aitrust_requests_total",
    "Total analyze requests",
    ["status", "risk_level"],
)

LATENCY = Histogram(
    "aitrust_latency_seconds",
    "Request latency",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

CONFIDENCE = Histogram(
    "aitrust_confidence",
    "Predicted confidence distribution",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

CASCADE_PATH = Counter(
    "aitrust_cascade_path",
    "Cascade path taken",
    ["path"],
)


def record_request(
    status: str,
    risk_level: str | None,
    confidence: float | None,
    latency_s: float,
    cascade_path: str | None = None,
) -> None:
    """Record metrics for a completed request."""
    REQUESTS_TOTAL.labels(status=status, risk_level=risk_level or "none").inc()
    LATENCY.observe(latency_s)
    if confidence is not None:
        CONFIDENCE.observe(confidence)
    if cascade_path:
        CASCADE_PATH.labels(path=cascade_path).inc()


# Middleware логирования запросов.


def _extract_correlation_id(request: Request) -> str:
    """Достать correlation_id из заголовков клиента или сгенерировать новый.

    Принимаем X-Correlation-ID и X-Request-ID; первый непустой и валидный UUID
    выигрывает. Невалидный UUID игнорируется и заменяется на сгенерированный
    UUID4 — это нужно потому что БД-колонка `analyses.request_id` имеет тип UUID,
    а correlation_id используется как request_id.
    """
    for header in (CORRELATION_ID_HEADER, *ALT_CORRELATION_HEADERS):
        candidate = request.headers.get(header)
        if not candidate:
            continue
        try:
            uuid.UUID(candidate)
            return candidate
        except ValueError:
            continue  # невалидный UUID — игнорируем, идём на следующий header
    return str(uuid.uuid4())


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Установить correlation_id в ContextVar, написать структурный лог /api/-запросов,
    вернуть correlation_id клиенту через заголовок ответа."""

    async def dispatch(self, request: Request, call_next):
        cid = _extract_correlation_id(request)
        set_correlation_id(cid)

        is_api = request.url.path.startswith("/api/")
        t0 = time.perf_counter()
        response: Response = await call_next(request)
        elapsed = time.perf_counter() - t0

        response.headers[CORRELATION_ID_HEADER] = cid

        if is_api:
            log.info(
                "request",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "status_code": response.status_code,
                    "latency_ms": round(elapsed * 1000, 2),
                },
            )
        return response


# Endpoint /metrics для Prometheus.


async def metrics_endpoint(request: Request):
    """GET /metrics — Prometheus scrape target."""
    return StarletteResponse(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
