# AI Text Trust Service

Сервисная система идентификации искусственно сгенерированных текстов для повышения доверия к информационному контенту. Принимает русскоязычный текст и возвращает вероятностную оценку того, что он написан языковой моделью, а также стилометрические маркеры, на которые опирается этот вывод, внутри несколько сменных детекторов под разные требования к скорости и точности.

> Прод-стенд: **https://89.169.141.35.sslip.io** · Swagger: `/docs` · Readiness: `/ready`

## Возможности

- **Три детектора** — `tfidf`, `transformer` (ruRoBERTa-large), `cascade`. Переключаются через `DETECTOR_TYPE` без пересборки образа.
- **Sync и async-режимы** — `/api/v1/analyze` для коротких запросов, `/api/v1/jobs` для каскада на CPU (БД-очередь через `FOR UPDATE SKIP LOCKED`, без Redis).
- **Batch-обработка** — загрузка CSV через `/api/v1/batch`, потоковая выгрузка результатов.
- **File extract** — DOCX / PDF / CSV → plain text для анализа.
- **Аутентификация** — JWT HS256 + opaque refresh-rotation, argon2id, slowapi rate-limit. Анонимный режим тоже поддерживается.
- **Explainability** — `?explain=true` возвращает стилометрические маркеры в сравнении с human-бейзлайном того же домена.
- **Наблюдаемость** — Prometheus `/metrics`, JSON-логи с `correlation_id`, `/health` и `/ready` для k8s-проб.
- **Web-клиент** — Next.js 16 / React 19 в [apps/web/](apps/web/), installable web app (manifest + standalone), деплоится same-origin рядом с API.
- **Приватность по умолчанию** — в БД хранится `sha256(text)`, не сам текст; `analyses.user_id` nullable для анонимов.

## Быстрый старт

```bash
uv sync                              # зависимости (main + dev + ml)
cp .env.example .env                 # настройки
make docker-db                       # Postgres в Docker
uv run alembic upgrade head          # миграции
make run                             # API на :8000
```

Web-клиент:

```bash
cd apps/web && npm install && npm run dev   # http://localhost:3000
```

Полностью в Docker — `make docker-up`. Подробно — [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Детекторы

| Режим | Латентность | Когда выбирать |
|---|---|---|
| `tfidf` *(по умолчанию)* | ~50 мс CPU | SLA важнее редких сложных случаев; устойчив к невидимым генераторам |
| `transformer` | ~500 мс GPU / 2–3 с CPU | максимум качества на знакомых генераторах |
| `cascade` | ~50 мс fast / 5–15 с slow | лучший средний результат; «серая зона» уходит в трансформер |

Пороги каскада подбираются под собранные артефакты: `make tune-cascade` пишет результат в `artifacts/cascade_threshold_sweep.json`. Подробности и обоснование признаков — [docs/DETECTORS.md](docs/DETECTORS.md).

## Команды

| Команда | Что делает |
|---|---|
| `make run` | API на `:8000` с auto-reload |
| `make docker-up` | API + Postgres в Docker |
| `make test` | unit + integration (без `data`-маркера) |
| `make test-all` | полный pytest, включая проверки датасета |
| `make benchmark` | бенчмарк латентности → `artifacts/latency_benchmark.json` |
| `make audit-splits` | отчёт по утечкам и n-граммам в сплитах |
| `make tune-cascade` | подбор порогов каскада на стратифицированной подвыборке |
| `make lint` / `make format` | ruff check / format |

## Документация

| Документ | О чём |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | слои `src/app/`, путь запроса, async-воркер, схема БД |
| [docs/DETECTORS.md](docs/DETECTORS.md) | TF-IDF, ruRoBERTa, каскад, артефакты, ограничения |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | локальный запуск и продакшн в Yandex Cloud |
| [docs/SECURITY.md](docs/SECURITY.md) | модель угроз, auth, rate-limit, приватность |
| [docs/ROADMAP.md](docs/ROADMAP.md) | известные ограничения MVP и приоритеты |

## Дисклеймер

Вердикт детектора — вероятностная оценка, а не факт. Сервис не проверяет фактологию текста и хуже работает на коротких фрагментах и постредактированном AI-тексте. Текст дисклеймера попадает в Swagger и в каждое поле `AnalyzeResponse.disclaimer`; переопределяется через `DISCLAIMER_TEXT`.

