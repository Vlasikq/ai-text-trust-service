"""FastAPI dependencies для auth.

Два dependency-режима:
  - `get_current_user_optional`: возвращает User или None — для эндпоинтов с двумя режимами
    (anonymous + authenticated, главный пример — POST /api/v1/analyze).
  - `get_current_user`: 401 если токен отсутствует / невалиден. Для /me, /me/analyses.

Извлечение токена — Authorization: Bearer <jwt>.
"""

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.tokens import TokenError, decode_access_token
from app.config import Settings, get_settings
from app.database.models import User, UserStatus
from app.database.uow import UnitOfWork

# auto_error=False → отсутствие заголовка не бросает 403, мы сами решаем.
_bearer_optional = HTTPBearer(auto_error=False)
_bearer_required = HTTPBearer(auto_error=True)


def _get_app_settings(request: Request) -> Settings:
    """Settings лежат на app.state — кладёт lifespan в main.py.

    Fallback на get_settings() — для случаев, когда зависимость дёргается из контекста
    без app (тесты, ручные вызовы).
    """
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        settings = get_settings()
    return settings


async def _resolve_user_from_token(token: str, settings: Settings) -> User | None:
    try:
        payload = decode_access_token(
            token,
            secret=settings.jwt_secret,
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
            algorithm=settings.jwt_algorithm,
        )
    except TokenError:
        return None

    async with UnitOfWork() as uow:
        user = await uow.users.get_by_id(payload.user_id)
        if user is None or user.status != UserStatus.ACTIVE:
            return None
        return user


async def get_current_user_optional(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_optional),
) -> User | None:
    """Вернуть User если bearer-токен валиден, иначе None (anonymous-режим)."""
    if credentials is None:
        return None
    settings = _get_app_settings(request)
    return await _resolve_user_from_token(credentials.credentials, settings)


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_required),
) -> User:
    """Строгая версия: 401 если токен отсутствует / невалиден / пользователь disabled."""
    settings = _get_app_settings(request)
    user = await _resolve_user_from_token(credentials.credentials, settings)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
