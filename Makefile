.PHONY: install run test openapi docker-up lint

install:
	python -m pip install -r requirements.txt

run:
	PYTHONPATH=src uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	PYTHONPATH=src pytest -q

openapi:
	PYTHONPATH=src python scripts/export_openapi.py

docker-up:
	docker compose -f docker/docker-compose.yml up --build

lint:
	ruff check .
