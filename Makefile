.PHONY: install run test openapi docker-up lint sync

sync:
	uv sync

run:
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	uv run pytest -q

openapi:
	uv run python scripts/export_openapi.py

docker-up:
	docker compose -f docker/docker-compose.yml up --build

lint:
	uv run ruff check .

format:
	uv run ruff format .
