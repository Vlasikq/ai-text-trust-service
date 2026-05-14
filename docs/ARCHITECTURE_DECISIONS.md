# Architecture Decision Records

Здесь фиксируются осознанные архитектурные решения, которые не выводятся из кода
напрямую и могут вызвать вопрос «почему именно так».

Формат основан на Michael Nygard, «Documenting Architecture Decisions» (2011):
**Context** → **Decision** → **Consequences**. Каждая ADR — иммутабельна: правки
оформляются как новый ADR со ссылкой на предыдущий.

---


### Context

В сервисе сейчас сосуществуют два префикса:

- **Без префикса**: `/health`, `/ready`, `/metrics`, `/auth/{register,login,refresh,logout}`,
  `/me`, `/me/analyses`, `/me/analyses/{id}`, `/me/stats`, `/me/batches/{id}`.
- **`/api/v1/`**: `/api/v1/analyze`, `/api/v1/batch`, `/api/v1/jobs/{id}`,
  `/api/v1/feedback`, `/api/v1/extract/text`.

Эта несимметрия видна в OpenAPI (`/openapi.json`) и в Caddyfile (где matcher
`@api` явно перечисляет оба набора). Соблазн «причесать всё под `/api/v1/`» есть,
но он стоит миграции для всех клиентов (PWA, Telegram-бот, smoke-скрипт) и
переписывания auth-cookie path (если когда-нибудь введём).

### Decision

Зафиксировать текущее разделение как осознанную политику:

- **Operational endpoints** (`/health`, `/ready`, `/metrics`) живут на корне без
  префикса. Это стандарт Kubernetes / Spring Boot Actuator / Vault: оркестратор
  и SRE-tooling ожидают health-probes по фиксированным путям. Версии API на них
  не действуют — операционный контракт не должен сломаться при v2.
- **Identity endpoints** (`/auth/*`, `/me`, `/me/*`) тоже без префикса. Это
  «пользовательский» слой, а не «доменный API»: в типичной REST-архитектуре auth
  и сессионные эндпоинты ортогональны бизнес-логике и редко версионируются вместе
  с ней (см. OAuth 2.0 RFC-6749, OpenID Connect Core — все без `/v1/`).
- **Domain endpoints** (`/api/v1/analyze`, `/batch`, `/jobs`, `/feedback`,
  `/extract`) идут под `/api/v1/`. Это бизнес-API, который может развиваться с
  ломающими изменениями (новый формат AnalyzeResponse, миграция на v2 контракта
  feedback) — версия в префиксе позволяет вводить v2 без disruption v1.

### Consequences

**Плюсы:**

- Operational пути стабильны: оркестратор не сломается при перевыпуске API.
- Auth/Identity не требует ребрендинга при v2 domain API.
- Caddyfile matcher `@api` остаётся коротким: явный allow-list путей FastAPI.

**Минусы:**

- Asymmetria видна при первом взгляде; нужно объяснять (этот ADR — и есть
  объяснение).
- Если решим унифицировать (всё под `/api/v1/`) — будет ломающее изменение
  для PWA, Telegram-бота и smoke-скрипта. Откладываем такую унификацию до
  необходимости (например, появления v2 контракта auth с
  ротацией refresh-токенов через cookie).

**Pre-conditions для миграции:**

- v2 контракт hado иметь объективные причины (новый формат TokenPair, BFF
  для refresh-cookie).
- Клиенты переходят на новый префикс параллельно (через feature flag в
  `NEXT_PUBLIC_API_URL`).
- Старый префикс оставляется как deprecated на ≥ 1 релиз, чтобы legacy-клиенты
  успели обновиться.
