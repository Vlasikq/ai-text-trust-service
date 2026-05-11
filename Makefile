.PHONY: install run test test-all openapi docker-up lint sync demo demo-offline benchmark export-artifacts \
	audit-splits repo-stats tune-cascade split-generators docker-image-info

export PYTHONPATH := src

sync:
	uv sync

run:
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Default: unit + integration (без data — нужен собранный датасет).
test:
	uv run pytest tests/unit tests/integration -q

# Только быстрые юниты (без БД / testcontainers) — для tight inner loop разработки.
test-fast:
	uv run pytest tests/unit -q

# Только data-тесты — нужны data/splits/*.jsonl (см. scripts/assemble_dataset.py).
test-data:
	uv run pytest tests/data -q

# Полный прогон: unit + integration + data.
test-all:
	uv run pytest -q

# Gradio UI → HTTP к FastAPI (подними API: make docker-up или make run)
demo:
	uv run python demo/app.py

# Тот же UI, но модели в процессе Gradio (без бэкенда на :8000)
demo-offline:
	uv run python demo/app.py --offline

benchmark:
	uv run python scripts/benchmark_latency.py

audit-splits:
	uv run python scripts/audit_splits.py

repo-stats:
	uv run python scripts/repository_stats.py

tune-cascade:
	uv run python scripts/tune_cascade_thresholds.py

split-generators:
	uv run python scripts/report_split_generators.py

docker-image-info:
	uv run python scripts/docker_image_info.py

export-artifacts:
	uv run python scripts/export_service_artifacts.py

openapi:
	uv run python scripts/export_openapi.py

docker-up:
	docker compose -f docker/docker-compose.yml up --build

docker-db:
	docker compose -f docker/docker-compose.yml up db -d

docker-down:
	docker compose -f docker/docker-compose.yml down

lint:
	uv run ruff check .

format:
	uv run ruff format .
