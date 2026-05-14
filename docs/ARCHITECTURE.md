# Архитектура сервиса

Сервис определяет, написан ли русскоязычный текст человеком или языковой моделью. Внешний контракт простой: `POST /api/v1/analyze` принимает текст и возвращает вероятностный вердикт `ai` или `human`, уровень риска и, по запросу, объяснение через стилометрические маркеры. Внутри живут три детектора (TF-IDF, ruRoBERTa-large, каскад) и единый конвейер обработки: предобработка, инференс, калибровка, формирование вердикта, объяснение. Запросы идут по двум путям — синхронный HTTP-вызов и асинхронный воркер для batch-обработки.

Решения, которые не выводятся напрямую из кода и могут вызвать вопрос «почему именно так», вынесены в [ARCHITECTURE_DECISIONS.md](ARCHITECTURE_DECISIONS.md) (ADR в формате Nygard). Сейчас зафиксирован один: ADR-001 объясняет политику префиксов API — operational и identity endpoints живут на корне, доменные — под `/api/v1/`.

## Стек

| Слой | Технология |
|---|---|
| API | FastAPI (lifespan-инициализация, async-роутеры) |
| ORM | SQLAlchemy 2 + asyncpg, миграции Alembic |
| Хранилище | PostgreSQL (managed в YC, локально в Docker) |
| Очередь | БД-очередь через `FOR UPDATE SKIP LOCKED` (без Redis/RabbitMQ) |
| Frontend | Next.js 16 (installable web app, output: standalone), React 19, Tailwind v4 |
| Reverse proxy | Caddy 2 (TLS-автоматизация через Let's Encrypt) |
| Аутентификация | JWT HS256 (access) + opaque refresh с ротацией |
| Наблюдаемость | Prometheus + structured JSON logging с correlation_id |
| Сборка | uv (lockfile), Docker multi-stage |

## Слои `src/app/`

```
src/app/
├── main.py             # FastAPI factory + lifespan: load detector → calibrator → explainer → cache → DB
├── worker.py           # асинхронный воркер для каскадных задач (тот же путь анализа, что и в HTTP)
├── config.py           # Pydantic Settings, кешированный get_settings()
├── schemas.py          # все Pydantic-модели запросов/ответов в одном файле
├── cache.py            # in-memory LRU (sha256(text) → DetectionResult)
├── logging_config.py   # JSON-логгер + ContextVar correlation_id
│
├── routers/            # HTTP-слой, тонкий
│   ├── health.py       # /health (liveness), /ready (readiness)
│   ├── auth.py         # /auth/{register,login,refresh,logout}
│   ├── me.py           # /me, /me/analyses, /me/analyses/{id}, /me/stats
│   ├── analyze.py      # /api/v1/analyze (+ /analyze/segments)
│   ├── jobs.py         # /api/v1/jobs (создание/опрос async-задач)
│   ├── batch.py        # /api/v1/batch (CSV-загрузка) + /me/batches
│   ├── extract.py      # /api/v1/extract/text (docx/pdf/csv → plain text)
│   └── feedback.py     # /api/v1/feedback
│
├── services/           # бизнес-логика, не зависит от HTTP/DB
│   ├── detection.py    # run_detection(ctx, text) — единый pipeline для sync и worker
│   ├── file_extractor.py
│   ├── csv_parser.py
│   └── segment_scoring.py  # per-sentence scoring для текста
│
├── detectors/          # ML inference, общий BaseDetector
│   ├── base.py         # BaseDetector + DetectionResult
│   ├── tfidf.py        # TF-IDF (char + word) + LogReg, ~50 мс
│   ├── transformer.py  # ruRoBERTa-large, ~500 мс GPU / 2-3 с CPU
│   └── cascade.py      # TF-IDF → если в серой зоне → трансформер
│
├── preprocessing/      # очистка markdown, нормализация пробелов, truncate
│   ├── pipeline.py     # TextPreprocessor: clean → length-check → warnings
│   └── text_cleaning.py  # clean_text + truncate_by_sentence (shared with assemble_dataset.py)
│
├── features/           # стилометрические признаки (45 фич) для explain=true
│   └── stylometric.py  # extract_all() → DataFrame с per-feature значениями
│
├── calibration/        # Platt scaling, ECE-проверка
│   └── platt.py
│
├── explanation/        # сравнение признаков с human-baselines → top markers
│   └── markers.py      # StyleExplainer: использует features.stylometric
│
├── middleware/
│   ├── ratelimit.py    # slowapi, key_func разделяет anon (IP) и auth (token-prefix)
│   └── metrics.py      # Prometheus counters/histograms + correlation_id middleware
│
├── auth/
│   ├── passwords.py    # argon2id + constant-time dummy verify для constant-time login
│   ├── tokens.py       # JWT HS256 + opaque refresh (sha256 в БД)
│   └── dependencies.py # get_current_user / get_current_user_optional
│
└── database/
    ├── models.py       # 6 ORM-моделей (Analysis, Feedback, AnalysisJob, BatchJob, User, RefreshToken, ModelDeployment)
    ├── engine.py       # init_engine / get_engine / dispose_engine
    ├── uow.py          # Unit of Work: один Session на запрос, репозитории внутри
    ├── repositories.py # AnalysisRepository, JobRepository, UserRepository, RefreshTokenRepository
    └── persist.py      # response_to_model — AnalyzeResponse → Analysis row
```

Роутеры намеренно тонкие: собирают входной payload, вызывают сервисный слой и возвращают Pydantic-модель. Вся логика анализа находится в `services/detection.py`, и поэтому синхронный endpoint и async-воркер ходят одним и тем же путём.

## Запрос `POST /api/v1/analyze`

```
┌────────┐
│ client │
└───┬────┘
    │ POST /api/v1/analyze {text, explain}
    ▼
┌─────────────────────────────────────┐
│ RequestLoggingMiddleware             │
│  • extract X-Correlation-ID → CtxVar │
│  • t0 = perf_counter()               │
└───┬─────────────────────────────────┘
    ▼
┌─────────────────────────────────────┐
│ slowapi @limiter.limit(60/minute)    │
│  key_func: auth:<token-prefix> | ip:<x>
└───┬─────────────────────────────────┘
    ▼
┌─────────────────────────────────────┐
│ get_current_user_optional (Depends)  │
│  no token → user=None (anon-режим)   │
└───┬─────────────────────────────────┘
    ▼
┌─────────────────────────────────────┐
│ run_detection(ctx, text)             │
│  1. preprocess: clean + truncate     │
│  2. cache lookup (sha256 → result)   │
│  3. detector.predict (TF-IDF/cascade)│
│  4. Platt calibrate (если ECE>0.05)  │
│  5. risk_level + verdict             │
│  6. explain (если explain=true)      │
└───┬─────────────────────────────────┘
    ▼
┌─────────────────────────────────────┐
│ persist Analysis (с fallback)        │
│  • text_hash, не raw text            │
│  • user_id NULL для анонимов         │
│  • при падении БД — fallback в JSONL │
└───┬─────────────────────────────────┘
    ▼
┌─────────────────────────────────────┐
│ record_request → Prometheus metrics  │
└───┬─────────────────────────────────┘
    ▼
  AnalyzeResponse (+ disclaimer)
```

Что важно знать про этот путь:

- Приватность по умолчанию. Сырой текст не сохраняется: в БД лежит только `sha256(cleaned_text)`, длина и метаданные детекции. В логи текст тоже не попадает, если `LOG_TEXT=false`.
- Анонимный режим. `/api/v1/analyze` работает без авторизации, и в этом случае `analyses.user_id` остаётся `NULL`.
- Запись результата без жёсткой связности с БД. Если БД упала, fallback-логгер дописывает анализ в `db_fallback.jsonl`, а сервис продолжает отвечать клиенту.
- Дисклеймер в каждом ответе. Текст берётся из `Settings.disclaimer_text` и переопределяется переменной окружения `DISCLAIMER_TEXT`.

## Async-задачи и batch-обработка

Когда каскад попадает в «серую зону» (TF-IDF выдал вероятность в диапазоне `0.30 ≤ p ≤ 0.70`), нужен трансформер, а он на CPU отвечает за 5–15 секунд. Под такие запросы синхронный HTTP не годится, и для них предусмотрен второй путь:

1. Клиент шлёт `POST /api/v1/jobs` и получает `job_id` плюс `poll_url`.
2. Запись `AnalysisJob` создаётся со статусом `PENDING`, поле `input_text` хранится до завершения работы и зануляется по окончании.
3. Воркер (`python -m app.worker`) выбирает работу запросом `SELECT ... FROM analysis_jobs WHERE status='PENDING' ORDER BY created_at LIMIT 1 FOR UPDATE SKIP LOCKED`. Так несколько воркеров могут конкурировать за очередь без Redis.
4. Воркер вызывает тот же `run_detection`, сохраняет результат в `Analysis` и переводит задачу в `SUCCESS`, проставляя `FK` на анализ.
5. Клиент периодически опрашивает `/api/v1/jobs/{id}` и, увидев `status=SUCCESS`, забирает результат.

Batch-режим (загрузка CSV) устроен похоже:

- `POST /api/v1/batch` принимает multipart-CSV. Файл парсится через `services/csv_parser.py` и проверяется по длинам.
- Создаётся одна запись `BatchJob` (родитель) и `N` записей `AnalysisJob` (дети, ссылающиеся на батч через `batch_id`).
- Partial unique index `uq_one_active_batch_per_user WHERE status IN ('PENDING','PROCESSING')` не даёт пользователю запустить второй активный батч.
- Счётчики `n_completed` и `n_failed` инкрементируются атомарным SQL-выражением, что защищает от гонки при `--scale worker=2`.
- Отчёт скачивается потоково через `GET /api/v1/batch/{id}/results.csv` или `.json`.

## Жизненный цикл приложения и DI

Точка входа — `@asynccontextmanager async def lifespan(app)` в [src/app/main.py:131](src/app/main.py#L131). Инициализация идёт строго в таком порядке:

1. `setup_logging(LOG_LEVEL)` подключает JSON-форматтер и `ContextVar` для `correlation_id`.
2. Прод-инварианты (`JWT_SECRET`, `DATABASE_URL`, `CORS_ORIGINS`) проверяет `@model_validator` в `Settings` — раньше lifespan, поэтому покрывает и `worker.py`. 
3. `app.state.preprocessor = TextPreprocessor(settings)`.
4. `app.state.detector = _build_detector(settings)` собирает TF-IDF, трансформер или каскад — подробности в `DETECTORS.md`.
5. `app.state.cache = DetectionCache(maxsize=...)`.
6. `app.state.calibrator = ProbabilityCalibrator(...)` — опционально, подключается, когда ECE превышает 0.05.
7. `app.state.explainer = StyleExplainer(baselines_path)` — опционально.
8. `_init_database` поднимает engine и пишет запись `ModelDeployment` со снапшотом конфигурации.
9. Прогрев: один вызов `predict("Тестовый текст...")` стабилизирует p95.

`Depends(get_settings)` возвращает синглтон, кэшируемый через `@lru_cache(maxsize=1)`. В тестах, где нужны нестандартные настройки, объект `Settings(...)` создают руками.

`UnitOfWork` — это асинхронный контекст вокруг `AsyncSession`. Внутри лежат репозитории по доменам: `uow.users`, `uow.refresh_tokens`, `uow.analyses`, `uow.jobs`, `uow.deployments`. Транзакция коммитится при выходе из `async with`, и откатывается, если внутри было исключение.

## Воркер

Воркер стартует параллельно с API командой `python -m app.worker`. Образ Docker один и тот же, отличается только запускаемая команда — в `scripts/deploy/docker-compose.prod.yml` указано `command: ["python", "-m", "app.worker"]`.

Основной цикл лежит в [src/app/worker.py:246](src/app/worker.py#L246):

```python
while not _shutdown_requested:
    processed = await process_one_job(ctx, deployment_id)
    if not processed:
        await asyncio.sleep(POLL_INTERVAL_S)  # 2.0 секунды
```

Функция `process_one_job` ([src/app/worker.py:155](src/app/worker.py#L155)) делает следующее:

1. `uow.jobs.claim_next(WORKER_ID)` атомарно забирает `PENDING`-задачу через `FOR UPDATE SKIP LOCKED`, переводит её в `PROCESSING`, проставляет `worker_id` и `started_at`.
2. Запускает `run_detection`. Таймаут не выставляется специально: каскад на CPU может работать дольше десяти секунд.
3. При статусе `SUCCESS` `response_to_model` сохраняет запись `Analysis`, после чего вызывается `mark_success(job, analysis_id)`.
4. При `NO_DECISION` или `ERROR` ставится `mark_error(job, error_message)`.
5. Если у задачи проставлен `batch_id`, счётчик у родительской записи `BatchJob` инкрементируется атомарно.

По `SIGTERM` или `SIGINT` воркер останавливается аккуратно: дорабатывает текущий job, новых задач не берёт и закрывает engine.

## Схема БД (6 таблиц)

```
users                       refresh_tokens
├── id (uuid7 pk)            ├── id (uuid7 pk)
├── email (unique)           ├── user_id → users.id
├── password_hash            ├── token_hash (sha256, unique)
├── role (user|admin)        ├── expires_at, revoked, revoked_at
├── status (active|disabled) ├── replaced_by_id → refresh_tokens.id
└── created_at, last_login   └── user_agent, ip

analyses                    feedback
├── id (uuid7 pk)            ├── id (uuid7 pk)
├── request_id (unique)      ├── analysis_id → analyses.id (SET NULL)
├── text_hash (NOT raw text) ├── user_verdict (human|ai)
├── verdict, confidence      ├── source (api|gradio|manual)
├── risk_level, detector_used└── comment, created_at
├── cascade_path, inference_ms
├── user_id → users.id      analysis_jobs (async-режим)
├── deployment_id → ...       ├── id (uuid7 pk)
├── warnings (jsonb)          ├── status (PENDING|PROCESSING|SUCCESS|ERROR)
├── explanation (jsonb)       ├── input_text (обнуляется после finish)
└── requested_at              ├── text_hash, text_length
                              ├── user_id, batch_id, analysis_id (FK)
model_deployments             └── created_at, started_at, finished_at
├── id (uuid7 pk)
├── service_version          batch_jobs
├── model_version            ├── id (uuid7 pk)
├── detector_type            ├── user_id (CASCADE)
├── config_snapshot (jsonb)  ├── n_total, n_completed, n_failed
└── deployed_at              ├── status (PENDING|PROCESSING|COMPLETED)
                             ├── skipped_summary (jsonb)
                             └── partial-unique: один активный батч на user
```

Что важно про схему БД:

- В качестве первичных ключей используется UUID v7. Он сортируется по времени, не фрагментирует B-tree и удобно ложится на cursor-пагинацию.
- `warnings` и `explanation` лежат в `JSONB`, а не разнесены по отдельным таблицам. Чтение почти всегда идёт «вся запись `analysis` целиком», поэтому декомпозиция не нужна.
- `analyses.user_id` сделан nullable: анонимные запросы не привязываются к пользователю.
- `analysis_jobs.input_text` зануляется по завершении задачи, чтобы текст не оставался в БД дольше необходимого.
- Partial unique index на `batch_jobs` гарантирует на уровне БД, что у пользователя не появится второго активного батча.

## Наблюдаемость

На `/metrics` отдаются четыре Prometheus-серии:

- `aitrust_requests_total{status, risk_level}` — counter с количеством обработанных запросов.
- `aitrust_latency_seconds` — histogram с бакетами от 10 мс до 5 с.
- `aitrust_confidence` — histogram с десятью бакетами от 0.1 до 1.0.
- `aitrust_cascade_path{path}` — counter путей каскада: `fast`, `slow`, `fallback`.

Логи пишутся в JSON через `python-json-logger`, и в каждую запись подмешивается `correlation_id` из `ContextVar`. Заголовок `X-Correlation-ID` (или `X-Request-ID`) принимается из запроса и возвращается в ответе. Клиент может склеить свои логи с серверными по этому полю. Если в запросе пришёл невалидный UUID, генерируется свежий UUID4 — поле `analyses.request_id` имеет тип UUID на уровне БД.

## Конфигурация

Все настройки собраны в `Settings(BaseSettings)` в [src/app/config.py:25](src/app/config.py#L25). Источников два: `.env` для локальной разработки и переменные окружения в проде. Для каждого параметра:

- задан дефолт в коде;
- значение описано в `.env.example`;
- переопределяется через ENV без пересборки образа.

`SERVICE_VERSION` должна совпадать в трёх местах: `pyproject.toml`, `Settings.service_version` и тег docker-образа. Когда они расходятся, Swagger UI и метаданные `Analysis.service_version` показывают одну версию, а образ — другую. Заметить такое на ревью сложно, и про эту ловушку отдельно написано в `SECURITY.md`.
