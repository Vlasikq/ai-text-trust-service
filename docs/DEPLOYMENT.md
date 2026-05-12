# Развёртывание

Документ описывает два сценария: локальную разработку через Docker Compose и продакшн на Yandex Cloud. В проде это одна Compute VM, managed PostgreSQL, Object Storage и Container Registry.

## Локальная разработка

### Требования
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) — менеджер зависимостей
- Docker + Docker Compose (для БД и e2e-тестов)

### Запуск без Docker

```bash
uv sync                            # установить зависимости
cp .env.example .env               # настройки по умолчанию
make docker-db                     # поднять только Postgres в Docker
uv run alembic upgrade head        # применить миграции
make run                           # API на :8000
```

Полезные адреса: `/docs` — Swagger UI, `/health` — liveness, `/ready` — readiness.

### Запуск через Docker Compose

```bash
make docker-up                     # сборка образа + API + Postgres
```

`docker/docker-compose.yml` поднимает два сервиса. Сервис `api` слушает порт 8000, собирается из `docker/Dockerfile` и монтирует `../artifacts` в режиме «только чтение». Сервис `db` слушает 5432 и работает на образе `postgres:16-alpine` с именованным volume `pgdata`.

Healthcheck у API сделан простым: `urllib.request.urlopen('http://localhost:8000/health')`.

### Тесты

```bash
make test          # сервис и unit-тесты (455+ тестов, ~25 секунд)
make test-all      # плюс @pytest.mark.data — проверки собранного датасета
```

Интеграционные тесты используют [testcontainers-postgres](https://testcontainers-python.readthedocs.io/) и поднимают свежий контейнер `postgres:16` на сессию тестов.

### Lint и форматирование

```bash
make lint          # ruff check
make format        # ruff format
```

Конфигурация — [pyproject.toml:75-83](pyproject.toml#L75-L83). Линт включает правила `E`, `F`, `W`, `I`. В ноутбуках игнорируем `E501` и `F841`.

## Продакшн: Yandex Cloud

### Архитектура

```
                Internet
                    │
                    ▼
          ┌───────────────────────┐
          │ Compute VM (single)   │
          │  s2.medium / 2 vCPU   │
          │  Caddy → 80/443 TLS   │
          │     │                 │
          │     ├─► api:8000      │ (FastAPI)
          │     │                 │
          │     ├─► worker        │ (cascade-jobs)
          │     │                 │
          │     ├─► web:3000      │ (Next.js standalone)
          │     │                 │
          │     └─► artifact-sync │ (one-shot, transformer 1.4 GB)
          └───────┬───────────────┘
                  │
        ┌─────────┼──────────────┬─────────────┐
        ▼         ▼              ▼             ▼
   Managed PG   Object Storage   Container     Lockbox
   (HA, ssl)   transformer.bin   Registry      (планируется)
                                 public-read
```

| Компонент | YC сервис | Назначение |
|---|---|---|
| Compute Instance | Compute Cloud | хост для контейнеров |
| База данных | Managed PostgreSQL 16 | реляционное хранилище с HA из коробки |
| Хранилище артефактов | Object Storage (S3) | чекпоинт трансформера ~1.4 ГБ |
| Образы | Container Registry | `aitrust-api`, `aitrust-web` |
| Сеть | VPC и публичный IP | DNS через sslip.io под Let's Encrypt |
| TLS | Caddy и Let's Encrypt | автоматический выпуск и обновление сертификатов |
| Секреты | env-файл в `/opt/aitrust/.env` | план — Lockbox после защиты ВКР |

### Маршрутизация (Caddy)

`scripts/deploy/Caddyfile`:
```caddy
__DOMAIN__ {
    @api path /api/* /auth/* /me /me/* /health /ready /metrics /docs /openapi.json /redoc
    handle @api {
        reverse_proxy api:8000
    }
    handle {
        reverse_proxy web:3000
    }
    @probes path /health /ready
    log_skip @probes
}
```

PWA и API живут на одном домене, и поэтому CORS-преамбулы не нужны, а refresh-куку в будущем можно будет ставить `SameSite=Strict`. Caddy явно перечисляет пути API, и всё остальное (`/`, `/login`, `/register`, `/me/history`, `/manifest.webmanifest`, статика `_next/*`) уходит в Next.js.

### Docker Compose в проде

В `scripts/deploy/docker-compose.prod.yml` поднято пять сервисов:

1. `api` — образ `aitrust-api:<version>`, переменные `IS_PRODUCTION=true` и `DETECTOR_TYPE=cascade`, `start_period: 60s` под lifespan, миграции и прогрев.
2. `worker` — тот же образ, только команда другая: `command: ["python", "-m", "app.worker"]`.
3. `artifact-sync` — разовый контейнер с `command: python /app/scripts/sync_artifacts.py`. Копирует трансформер из S3 в общий volume и завершается. `api` и `worker` ждут его `service_completed_successfully`.
4. `web` — образ `aitrust-web:<version>` с Next.js в standalone-режиме, переменная `NEXT_PUBLIC_API_URL=""` для same-origin.
5. `caddy` — образ `caddy:2-alpine`, читает `/opt/aitrust/Caddyfile`, использует именованные volume `caddy-data` (там лежат сертификаты Let's Encrypt) и `caddy-config`.

### Cloud-init и bootstrap

`scripts/deploy/cloud-init.yaml.template` — шаблон cloud-init. В нём четыре плейсхолдера:

- `__DOMAIN__` подставляется как `<VM_IP>.sslip.io` (бесплатный wildcard DNS);
- `__CADDYFILE_CONTENT__` — содержимое Caddyfile, инлайн;
- `__COMPOSE_CONTENT__` — содержимое `docker-compose.prod.yml`;
- `__BOOTSTRAP_CONTENT__` — bootstrap-скрипт.

`scripts/deploy/render_cloud_init.py` собирает `cloud-init.rendered.yaml` с подставленными значениями. Этот файл закрыт `.gitignore`, потому что в нём лежит `DATABASE_URL` с паролем.

`scripts/deploy/bootstrap.sh` запускается на виртуалке из секции `runcmd` и делает следующее:

1. Скачивает YC root CA, чтобы SSL-соединение с managed Postgres проходило проверку.
2. Прописывает `DOMAIN=<vm_ip>.sslip.io` в `/opt/aitrust/.env`.
3. Ждёт готовности docker daemon, потому что `apt`-установка идёт асинхронно.
4. Поднимает стек командой `docker compose up -d`.
5. Прогоняет healthcheck API из самой виртуалки: до 10 попыток с интервалом 15 секунд, потому что миграции и прогрев занимают около минуты.

Лог bootstrap пишется в `/var/log/aitrust-bootstrap.log` и в serial console, чтобы можно было отлаживать cloud-init без SSH.

### Деплой новой версии

1. Локально соберите и запушьте образ:

   ```bash
   docker build -f docker/Dockerfile -t cr.yandex/<registry-id>/aitrust-api:<version> .
   docker push cr.yandex/<registry-id>/aitrust-api:<version>
   ```

2. Обновите тег в строке `image: ...:<version>` в `scripts/deploy/docker-compose.prod.yml`.
3. Поменяйте переменную `SERVICE_VERSION` в том же файле — она должна совпадать с тегом образа.
4. Если поменялась схема БД, создайте миграцию: `uv run alembic revision --autogenerate -m "..."`.
5. Срендерите cloud-init: `uv run python scripts/deploy/render_cloud_init.py`.
6. Примените на виртуалке одним из двух способов:
   - пересоздать инстанс командой `yc compute instance create --metadata-from-file user-data=...`;
   - зайти по SSH (`ssh ubuntu@<ip>`) и выполнить `cd /opt/aitrust && docker compose pull && docker compose up -d`.

В CMD контейнера API при старте вызывается `alembic upgrade head`, и миграции применяются автоматически. Если миграция упала, контейнер не поднимется. Это сделано сознательно: на устаревшей схеме работать нельзя.

### Артефакты (трансформер)

Трансформер размером около 1.4 ГБ в образ не зашит. Так образ остаётся в районе 330 МБ, его быстрее pull-ать при масштабировании, и чекпоинт обновляется независимо от кода.

Контейнер `artifact-sync` запускает `scripts/sync_artifacts.py` от root и делает три шага:

1. Проверяет именованный volume `transformer-artifacts`.
2. Сравнивает содержимое с S3 по хэшу; если совпадает, ничего не делает.
3. Иначе скачивает `transformer/model/*` из бакета `aitrust-artifacts` в `/app/artifacts/transformer/model/`.

Чтобы загрузить новый чекпоинт в S3, используется `uv run python scripts/deploy/upload_transformer.py`.

### База данных

Managed PostgreSQL в YC подключается по такой строке:

```
DATABASE_URL=postgresql+asyncpg://aitrust:<pwd>@rc1a-...mdb.yandexcloud.net:6432/aitrust?ssl=require
```

Порт 6432 — это PgBouncer. SSL включён через `ssl=require` на стороне asyncpg, корневой сертификат YC лежит в системном trust store через `root.crt`.

Миграции лежат в `alembic/versions/`:

- `001_initial.py` создаёт таблицы `analyses`, `feedback`, `model_deployments`;
- `002_analysis_jobs.py` добавляет async-задачи;
- `003_users.py` — таблицы `users` и `refresh_tokens`;
- `004_batch_jobs.py` — batch-задачи и partial unique index.

Новая миграция создаётся так:

```bash
uv run alembic revision --autogenerate -m "description"
# проверить файл в alembic/versions/, поправить, если autogenerate ошибся
uv run alembic upgrade head      # применить локально
```

Откат — `uv run alembic downgrade -1`. В проде откатывать не принято: миграции должны быть аддитивными.

## Переменные окружения

Полный список с дефолтами лежит в `.env.example`. Ниже — те, что обязательно переопределяются в проде:

| ENV | Dev (.env.example) | Prod | Зачем |
|---|---|---|---|
| `SERVICE_VERSION` | из `pyproject.toml` | равен тегу образа | попадает в `/health`, Swagger, `Analysis.service_version` |
| `IS_PRODUCTION` | `false` | `true` | включает жёсткие проверки в lifespan |
| `DETECTOR_TYPE` | `tfidf` | `cascade` | гибрид TF-IDF и трансформера |
| `JWT_SECRET` | дефолтный | случайные 32+ байт | подписывает access-токены; с дефолтом сервис не стартует, если `IS_PRODUCTION=true` |
| `DATABASE_URL` | localhost | managed PG | строка с паролем |
| `CORS_ORIGINS` | пусто | пусто (same-origin) или `https://app.example.com` | нужно, если PWA живёт на отдельном домене |
| `LOG_LEVEL` | `INFO` | `INFO` | `DEBUG` — только под траблшутинг |
| `LOG_TEXT` | `false` | `false` | в проде никогда не `true` |

Ключи `OPENAI_API_KEY`, `GIGACHAT_*`, `YANDEX_API_KEY`, `GEMINI_API_KEY` нужны только скриптам сбора датасета в `scripts/data_collection/`. Runtime сервиса их не использует, и на проде их быть не должно.

## Пробы для оркестратора

`/health` — liveness-проба. Возвращает 200, пока процесс жив. В ответе два поля: `status` и `service_version`. Пример `livenessProbe` для k8s:
```yaml
livenessProbe:
  httpGet: { path: /health, port: 8000 }
  periodSeconds: 10
```

`/ready` — readiness-проба. Возвращает 503, если детектор ещё не загружен или если `DB_ENABLED=true`, а БД не отвечает на `SELECT 1`.

Поля ответа:
```json
{
  "status": "ready",
  "service_version": "0.4.0",
  "model_version": "ru-detector-0.1.0",
  "components": {
    "detector": {"ready": true, "info": {...}},
    "db": {"enabled": true, "ready": true},
    "calibrator": {"ready": false},
    "explainer": {"ready": true},
    "cache": {"size": 12, "maxsize": 1024, "hits": 7, "misses": 12, "hit_rate": 0.3684}
  }
}
```

Пример `readinessProbe` для k8s:
```yaml
readinessProbe:
  httpGet: { path: /ready, port: 8000 }
  periodSeconds: 5
  failureThreshold: 3
```

## Мониторинг

Prometheus scrape доступен на `/metrics`, всего четыре серии (подробности — в `ARCHITECTURE.md`, раздел «Наблюдаемость»). В проде на YC scrape пока не настроен, и это задача после защиты ВКР (см. `ROADMAP.md`). Локально проверить можно так:

```bash
make docker-up
curl http://localhost:8000/metrics
```

## Эксплуатационные нюансы

### Холодный старт около 60 секунд

В lifespan загружается трансформер (1.4 ГБ), затем применяется `alembic upgrade head` и идёт прогрев. `start_period: 60s` в healthcheck это учитывает. При масштабировании об этом нужно помнить, чтобы не уронить SLI.

### Один образ для API и воркера

`aitrust-api:<version>` — общий образ. Отличается только команда запуска: `uvicorn` против `python -m app.worker`. Так проще деплоить и есть гарантия, что путь анализа в обоих режимах одинаковый.

### Container Registry открыт на чтение

YC CR настроен как public-read через `system:allUsers / images.puller`. Это упрощение MVP, подробности — в `SECURITY.md`, раздел «Известные ограничения».

### Debug-SSH в cloud-init

В `scripts/deploy/cloud-init.yaml.template:7-14` лежит публичный SSH-ключ для отладки. Это нужно во время разработки; после защиты ВКР секция уберётся.

## Шпаргалка по Makefile

```
make sync             # uv sync — установка зависимостей
make run              # API на :8000 (uvicorn --reload)
make test             # service- и unit-тесты
make test-all         # дополнительно прогон по @pytest.mark.data
make lint             # ruff check
make format           # ruff format
make docker-up        # API и Postgres
make docker-db        # только Postgres
make docker-down      # остановка стека
make openapi          # сгенерировать openapi/openapi.yaml
make benchmark        # бенчмарк латентности, отчёт в artifacts/latency_benchmark.json
make audit-splits     # дедупликация и аудит пересечений сплитов
make tune-cascade     # подобрать (cascade_lo, cascade_hi) на holdout
make repo-stats       # число тестов и LOC src/app — artifacts/repository_stats.json
make split-generators # генераторы по сплитам — artifacts/split_generators.json
make docker-image-info # размеры образов — artifacts/docker_image_info.json
make export-artifacts # собрать service/ из artifacts/
make demo             # gradio-демо (требует поднятый API на 8000)
make demo-offline     # gradio-демо с инлайн-загрузкой моделей
```

## Что чаще всего ломается

| Симптом | Причина | Что делать |
|---|---|---|
| `/ready` отдаёт 503 с `detector.ready=false` | артефакты не загрузились или трансформер не подъехал | проверить `artifacts/service/` (TF-IDF) и `artifacts/transformer/model/`; посмотреть логи lifespan |
| При старте падает с «JWT_SECRET не переопределён» | переменная не задана | для `IS_PRODUCTION=true` нужно задать `JWT_SECRET=<random>`; локально хватает `IS_PRODUCTION=false` |
| Alembic падает с `connection refused` | БД ещё не поднялась | в compose помогает `depends_on: db: condition: service_healthy` |
| `409 Email already registered` сразу при первом `register` | пользователь уже есть в БД | проверить миграции: возможно, они применились на чужом окружении |
| После долгого ожидания каждый `/me` отдаёт 401 | refresh-токен протух (30 дней) | пользователю надо залогиниться заново |
| Предупреждение `LOW_CONFIDENCE` приходит почти на каждый ответ | модель близка к 0.5 | проверить артефакты и калибровку, пересобрать через `make export-artifacts` |
| Латентность каскада растёт | много запросов уходит на медленный путь | посмотреть метрику `aitrust_cascade_path`; если `slow` больше 20%, перенастроить пороги через `make tune-cascade` |
