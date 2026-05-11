# ── Stage 1: install dependencies via uv ──────────────────────
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

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
FROM python:3.11-slim

# Создаём non-root юзера ДО копирования — потом COPY --chown пишет сразу с правильным ownership.
# Иначе `chown -R` после COPY дублирует все 6+ GB venv в overlay-слое.
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Виртуальное окружение из builder-стадии
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Application code
COPY --chown=appuser:appuser src /app/src
COPY --chown=appuser:appuser scripts /app/scripts
COPY --chown=appuser:appuser alembic /app/alembic
COPY --chown=appuser:appuser alembic.ini /app/alembic.ini
COPY --chown=appuser:appuser .env.example /app/.env.example

# ML-артефакты для дефолтного DETECTOR_TYPE=tfidf (~2.5 MB).
# Трансформер не кладём в образ — для cascade подсасывается с S3 при старте.
COPY --chown=appuser:appuser artifacts/service /app/artifacts/service

USER appuser
ENV PYTHONPATH=/app/src:/app
EXPOSE 8000

# В production: alembic upgrade head перед стартом uvicorn. Если БД недоступна,
# alembic упадёт — это намеренно (не запускаемся без миграций).
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
