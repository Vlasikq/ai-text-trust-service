# aitrust-web

Next.js 16 (App Router) PWA для сервиса ai-text-trust. Это тонкий клиент над
FastAPI-бэком: форма анализа текста, история анализов, статистика, batch.

## Стек

- Next.js 16 App Router + React 19
- Tailwind v4 (через `@import "tailwindcss"`, без `tailwind.config`)
- recharts 3 для графиков на `/me/stats` (ленивый импорт)
- react-hook-form + zod для форм авторизации
- standalone-output Next.js → ~150 MB Docker-образ

## Команды

```powershell
npm ci                  # установить зависимости
npm run dev             # dev-сервер на :3000
npm run build           # production build
npm start               # запустить production build
npm run lint            # ESLint (eslint-config-next 9 flat config)
```

## ENV

| Переменная | Дефолт | Назначение |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `""` (same-origin) | Базовый URL API. Пустая строка = тот же origin, что и фронт (продовый паттерн с Caddy). Cross-origin: `https://api.example.com`. |

`.env.local` подхватывается `next` автоматически — там же кладите локальные
переопределения для dev.

## Где что лежит

- `src/app/` — App Router маршруты (главная, `/login`, `/register`, `/me/*`, `/batch`)
- `src/components/` — `AnalyzeForm`, `ResultCard`, `TextHighlights`, `SegmentedText`, `Nav`, `AuthForm`
- `src/lib/api.ts` — типы и обёртка fetch с авто-refresh на 401
- `src/lib/auth.tsx` — `AuthProvider` + `useAuth` хук
- `src/lib/labels.ts` — общие подписи и цвета verdict/risk/warning

## Контракт с бэком

API-эндпоинты, которые трогает фронт, перечислены в `apiEndpoints` в
[src/lib/api.ts](src/lib/api.ts). Префиксы — по политике из
[../../docs/ARCHITECTURE_DECISIONS.md](../../docs/ARCHITECTURE_DECISIONS.md)
(`/api/v1/*` — домен, `/auth/*` и `/me/*` — без префикса).

## Docker

Multi-stage `Dockerfile` собирает standalone-output (~150 MB). См.
[../../docs/DEPLOYMENT.md](../../docs/DEPLOYMENT.md) для прод-деплоя.

## Подробнее

- Корневой [README](../../README.md) — общая картина сервиса
- [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md) — бэкенд и слои
- [docs/DEPLOYMENT.md](../../docs/DEPLOYMENT.md) — Caddy, YC, cloud-init
