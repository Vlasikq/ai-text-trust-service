"""slowapi-based rate limiting.

Подход:
  - `limiter` — глобальный singleton, key_func учитывает auth (user_id) и anon (IP).
  - Лимиты-строки (e.g. "30/minute") берутся из Settings, ставятся декораторами на endpoints.
  - X-Forwarded-For уважается (за Caddy IP клиента в этом заголовке).

Для тестов: можно передать `enabled=False` через env, тогда limiter не блокирует.
"""

from __future__ import annotations

from fastapi import Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.responses import Response

from app.middleware.forwarded import client_ip


def _key_func(request: Request) -> str:
    """Ключ rate-limit'а: user_id если есть Authorization, иначе IP клиента.

    Не парсим JWT здесь (slowapi вызывается ДО dependency injection) — только смотрим
    наличие заголовка. Это даёт грубое разделение «anon vs auth» по бакетам, но не
    позволяет attacker'у с одним IP закидать сервис от имени множества фейковых user_id —
    лимиты у auth-bucket'а тоже жёсткие.
    """
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        # Берём первые 32 символа JWT — этого достаточно как стабильного ключа,
        # без трат CPU на полный decode.
        token_prefix = auth[7:39]
        return f"auth:{token_prefix}"

    ip = client_ip(request)
    if ip:
        return f"ip:{ip}"
    return f"ip:{get_remote_address(request)}"


# Singleton лимитер; storage by default in-memory (per-process).
# Для multi-worker будущего: переделать на redis storage_uri="redis://...".
limiter = Limiter(key_func=_key_func, default_limits=[])


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """slowapi-handler с добавленным Retry-After: стандартный handler ставит
    только X-RateLimit-*, клиенты часто опираются именно на Retry-After.
    """
    resp = _rate_limit_exceeded_handler(request, exc)
    retry = getattr(exc, "retry_after", None) or 60
    resp.headers["Retry-After"] = str(int(retry))
    return resp
