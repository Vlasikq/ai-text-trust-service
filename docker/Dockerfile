# ── Stage 1: install dependencies via uv ──────────────────────
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

# Поведение Python в контейнере: без .pyc и с unbuffered stdout/stderr —
# логи появляются в `docker logs` сразу, образ не растёт от __pycache__.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml uv.lock ./
# Группы зависимостей (см. pyproject.toml):
#   main (по дефолту) — FastAPI + sklearn + DB + auth + file-extract + boto3 + pandas
#   ml                — torch + transformers (для DETECTOR_TYPE=cascade/transformer)
# notebooks/data/demo не нужны в проде — не подсасываем.
# Для tfidf-only образа можно собрать без `--group ml` (–1.5 GB).
ARG INCLUDE_ML=1
RUN if [ "$INCLUDE_ML" = "1" ]; then \
        uv sync --frozen --no-dev --group ml --no-install-project; \
    else \
        uv sync --frozen --no-dev --no-install-project; \
    fi

# ── Stage 2: application ──────────────────────────────────────
# Пин на slim-bookworm: latest `slim` плавал между Debian releases и иногда
# перестраивал слои без видимого изменения тега.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# tini как init-PID1: пробрасывает SIGTERM uvicorn'у и reaps зомби.
# Без него PID1 (uvicorn или sh) игнорирует unhandled сигналы — `docker stop`
# ждёт 10 секунд и шлёт SIGKILL, graceful shutdown не работает.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/*

# Создаём non-root юзера ДО копирования — потом COPY --chown пишет сразу с правильным ownership.
# Иначе `chown -R` после COPY дублирует все 6+ GB venv в overlay-слое.
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Виртуальное окружение из builder-стадии
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Application code
COPY --chown=appuser:appuser src /app/src
# Из scripts/ нужен только sync_artifacts.py — его запускает artifact-sync
# контейнер тем же образом. Остальные скрипты — local CLI tooling.
COPY --chown=appuser:appuser scripts/sync_artifacts.py /app/scripts/sync_artifacts.py
COPY --chown=appuser:appuser alembic /app/alembic
COPY --chown=appuser:appuser alembic.ini /app/alembic.ini
COPY --chown=appuser:appuser pyproject.toml /app/pyproject.toml

# ML-артефакты для дефолтного DETECTOR_TYPE=tfidf (~2.5 MB).
# Трансформер не кладём в образ — для cascade подсасывается с S3 при старте.
COPY --chown=appuser:appuser artifacts/service /app/artifacts/service

USER appuser
ENV PYTHONPATH=/app/src:/app
EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--"]

# В production: alembic upgrade head перед стартом uvicorn. Если БД недоступна,
# alembic упадёт — это намеренно (не запускаемся без миграций).
# exec заменяет sh на uvicorn, чтобы сигналы от tini шли uvicorn напрямую.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
