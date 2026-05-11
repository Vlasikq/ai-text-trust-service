"""End-to-end smoke для production API.

Запуск:
    uv run python scripts/deploy/smoke_prod.py https://89.169.141.35.sslip.io/
    # или с .env.smoke (gitignored):
    SMOKE_USER_EMAIL=smoke@example.com SMOKE_USER_PASSWORD=... uv run python scripts/deploy/smoke_prod.py https://...

Что проверяет:
    1. GET  /health                 → 200, {"status": "ok"}
    2. POST /auth/register          → 201 (или 409 если уже есть)
    3. POST /auth/login             → 200, access_token + refresh_token
    4. GET  /me                     → 200, email совпадает
    5. POST /api/v1/analyze         → 200, verdict в [None, "ai", "human"]
    6. POST /auth/refresh           → 200, новые токены (старый refresh revoked)
    7. POST /auth/logout            → 204
    8. POST /auth/refresh (старый)  → 401 (refresh-reuse detection)

Exit code 0 при успехе, 1 при первой ошибке. Подробный лог в stderr.

Не использует jq и bash — портативно (Windows / Linux / macOS) при наличии httpx.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from typing import Any

try:
    import httpx
except ImportError:
    print("ERROR: pip install httpx (или uv run python ...)", file=sys.stderr)
    sys.exit(1)


TIMEOUT_SECONDS = 30.0
# Образец текста, который должен проходить минимальную длину анализатора (>=300 знаков).
SMOKE_TEXT = (
    "Современные нейронные сети демонстрируют впечатляющие результаты в задачах "
    "обработки естественного языка. Особенно заметен прогресс в области генерации "
    "текстов: модели уровня GPT-4 способны создавать связные осмысленные фрагменты, "
    "стилистически близкие к человеческой речи. Это поднимает важный вопрос "
    "идентификации авторства, особенно в академической среде, где использование "
    "сгенерированных текстов без указания источника считается недобросовестной "
    "практикой и нарушением академической этики."
)


class SmokeError(RuntimeError):
    pass


def _step(n: int, name: str) -> None:
    print(f"[{n}/8] {name}...", file=sys.stderr)


def _ok(msg: str) -> None:
    print(f"     OK — {msg}", file=sys.stderr)


def _fail(msg: str) -> None:
    raise SmokeError(msg)


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        _fail(msg)


def _expect(resp: httpx.Response, expected_status: int | tuple[int, ...]) -> dict[str, Any]:
    expected = (expected_status,) if isinstance(expected_status, int) else expected_status
    if resp.status_code not in expected:
        _fail(
            f"{resp.request.method} {resp.request.url.path} → {resp.status_code} "
            f"(ожидали {expected}); body: {resp.text[:300]}"
        )
    try:
        return resp.json() if resp.text else {}
    except ValueError:
        return {}


def run_smoke(base_url: str, email: str, password: str) -> None:
    base = base_url.rstrip("/")

    with httpx.Client(base_url=base, timeout=TIMEOUT_SECONDS, follow_redirects=False) as c:
        # 1. health
        _step(1, "GET /health")
        body = _expect(c.get("/health"), 200)
        _assert(body.get("status") == "ok", f"/health.status != 'ok': {body}")
        _ok("status=ok")

        # 2. register (201 при первом запуске smoke на новой VM; 409 если email занят)
        _step(2, f"POST /auth/register {email}")
        resp = c.post(
            "/auth/register",
            json={"email": email, "password": password},
        )
        if resp.status_code == 201:
            _ok("registered (201)")
        elif resp.status_code == 409:
            _ok("уже зарегистрирован (409, идём в login)")
        else:
            _fail(f"register → {resp.status_code}; body: {resp.text[:300]}")

        # 3. login
        _step(3, "POST /auth/login")
        body = _expect(
            c.post("/auth/login", json={"email": email, "password": password}),
            200,
        )
        access = body.get("access_token")
        refresh = body.get("refresh_token")
        _assert(bool(access) and bool(refresh), "login без токенов")
        _ok("access + refresh получены")

        auth_h = {"Authorization": f"Bearer {access}"}

        # 4. /me
        _step(4, "GET /me")
        body = _expect(c.get("/me", headers=auth_h), 200)
        _assert(body.get("email") == email, f"/me.email != {email}: {body}")
        _ok(f"email подтверждён ({body.get('email')})")

        # 5. /api/v1/analyze
        _step(5, "POST /api/v1/analyze")
        body = _expect(
            c.post(
                "/api/v1/analyze",
                json={"text": SMOKE_TEXT, "explain": False},
                headers=auth_h,
            ),
            200,
        )
        status = body.get("status")
        # Status enum value: "SUCCESS" | "NO_DECISION" | "ERROR"
        _assert(
            status in ("SUCCESS", "NO_DECISION"),
            f"analyze.status неожиданный: {status!r}",
        )
        verdict = body.get("verdict")
        _assert(
            verdict in (None, "ai", "human"),
            f"verdict неожиданный: {verdict!r} (ожидали None|'ai'|'human')",
        )
        confidence = body.get("confidence")
        _assert(
            confidence is None or (isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0),
            f"confidence вне [0,1]: {confidence!r}",
        )
        _assert(isinstance(body.get("text_length"), int), "text_length отсутствует/не int")
        conf_str = "None" if confidence is None else f"{confidence:.3f}"
        _ok(f"status=success, verdict={verdict!r}, confidence={conf_str}")

        # 6. refresh — новые токены, старый revoked
        _step(6, "POST /auth/refresh")
        body = _expect(c.post("/auth/refresh", json={"refresh_token": refresh}), 200)
        new_access = body.get("access_token")
        new_refresh = body.get("refresh_token")
        _assert(bool(new_access) and bool(new_refresh), "refresh без новых токенов")
        _assert(new_refresh != refresh, "refresh не ротирован")
        _ok("новая пара токенов получена, refresh ротирован")

        # 7. logout — revoke свежего refresh
        _step(7, "POST /auth/logout")
        resp = c.post("/auth/logout", json={"refresh_token": new_refresh})
        # logout идемпотентен — допускаем 200 или 204.
        _assert(resp.status_code in (200, 204), f"logout → {resp.status_code}")
        _ok(f"logout {resp.status_code}")

        # 8. reuse старого refresh — должен 401 (reuse-detection после refresh выше)
        _step(8, "POST /auth/refresh (reuse старого) → ожидаем 401")
        resp = c.post("/auth/refresh", json={"refresh_token": refresh})
        _assert(
            resp.status_code == 401,
            f"старый refresh принят ({resp.status_code}) — reuse-detection не сработал",
        )
        _ok("reuse старого refresh корректно отклонён (401)")


def _load_smoke_env() -> tuple[str, str]:
    """SMOKE_USER_EMAIL / SMOKE_USER_PASSWORD из env или дефолт.

    Для одиночного запуска: если переменные не заданы, используем уникальный email
    через uuid — каждый прогон создаёт нового пользователя. Это намеренно для smoke,
    чтобы не зависеть от внешнего состояния БД. Очистка таких аккаунтов — задача
    отдельного maintenance-скрипта (вне scope smoke).
    """
    email = os.environ.get("SMOKE_USER_EMAIL")
    password = os.environ.get("SMOKE_USER_PASSWORD")
    if not email:
        email = f"smoke+{uuid.uuid4().hex[:8]}@example.com"
    if not password:
        # Стабильный пароль для повторного login при существующем email,
        # но в этом сценарии email каждый раз новый — пароль одноразовый.
        password = "Smoke!Test#" + uuid.uuid4().hex[:16]
    return email, password


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "Usage: smoke_prod.py <BASE_URL>\n"
            "  Example: smoke_prod.py https://89.169.141.35.sslip.io/",
            file=sys.stderr,
        )
        return 1

    base_url = sys.argv[1]
    email, password = _load_smoke_env()

    print(f"=== SMOKE PROD: {base_url}", file=sys.stderr)
    print(f"=== User: {email}", file=sys.stderr)

    t0 = time.perf_counter()
    try:
        run_smoke(base_url, email, password)
    except SmokeError as e:
        elapsed = time.perf_counter() - t0
        print(f"\nFAIL после {elapsed:.1f}s:\n  {e}", file=sys.stderr)
        return 1
    except httpx.RequestError as e:
        elapsed = time.perf_counter() - t0
        print(f"\nFAIL (network) после {elapsed:.1f}s:\n  {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - t0
    print(f"\nOK — 8 шагов за {elapsed:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
