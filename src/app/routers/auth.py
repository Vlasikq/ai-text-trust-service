"""Endpoint'ы авторизации: register, login, refresh, logout.

Контракт:
  - `POST /auth/register` — создаёт пользователя и возвращает пару (access, refresh);
  - `POST /auth/login` — проверяет пароль через argon2 и возвращает ту же пару;
  - `POST /auth/refresh` — ротирует refresh-токен: старый помечает отозванным, выдаёт новый;
  - `POST /auth/logout` — отзывает предъявленный refresh-токен.

Что важно по части приватности и безопасности:
  - в БД хранится только `sha256(refresh)`; plain-токен клиент видит ровно один раз;
  - если пришёл уже отозванный refresh, отзываются все токены в цепочке пользователя;
  - на все ошибки login и refresh отвечаем 401 с одинаковым сообщением, чтобы не выдавать причину.
"""

import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.exc import IntegrityError

from app.auth.passwords import (
    burn_dummy_verify_async,
    hash_password_async,
    verify_password_async,
)
from app.auth.tokens import (
    create_access_token,
    create_refresh_token,
    hash_refresh_token,
)
from app.config import Settings, get_settings
from app.database.models import RefreshToken, User
from app.database.uow import UnitOfWork
from app.middleware.forwarded import client_ip as _client_ip
from app.middleware.ratelimit import limiter
from app.schemas import (
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    RegisterResponse,
    TokenPair,
    UserPublic,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


# Вспомогательные функции.


def _user_to_public(user: User) -> UserPublic:
    return UserPublic(
        id=str(user.id),
        email=user.email,
        role=user.role,
        status=user.status,
        created_at=user.created_at.isoformat() if user.created_at else "",
        last_login_at=user.last_login_at.isoformat() if user.last_login_at else None,
    )


def _user_agent(request: Request) -> str | None:
    ua = request.headers.get("User-Agent")
    if ua and len(ua) > 255:
        return ua[:255]
    return ua


async def _issue_token_pair(
    *,
    user: User,
    settings: Settings,
    request: Request,
    uow: UnitOfWork,
    parent: RefreshToken | None = None,
) -> tuple[TokenPair, RefreshToken]:
    """Сгенерировать access + refresh, refresh сохранить в БД (sha256)."""
    access = create_access_token(
        user_id=str(user.id),
        role=user.role,
        secret=settings.jwt_secret,
        issuer=settings.jwt_issuer,
        audience=settings.jwt_audience,
        algorithm=settings.jwt_algorithm,
        ttl_minutes=settings.access_token_ttl_minutes,
    )
    refresh_plain = create_refresh_token()
    refresh_hash = hash_refresh_token(refresh_plain)
    expires_at = datetime.now(timezone.utc) + timedelta(days=settings.refresh_token_ttl_days)

    refresh_row = RefreshToken(
        user_id=user.id,
        token_hash=refresh_hash,
        expires_at=expires_at,
        user_agent=_user_agent(request),
        ip=_client_ip(request),
    )
    saved = await uow.refresh_tokens.create(refresh_row)
    if parent is not None:
        await uow.refresh_tokens.revoke(parent, replaced_by=saved)

    return (
        TokenPair(
            access_token=access,
            refresh_token=refresh_plain,
            access_expires_in=settings.access_token_ttl_minutes * 60,
            refresh_expires_in=settings.refresh_token_ttl_days * 86400,
        ),
        saved,
    )


# Endpoint'ы.


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Регистрация нового пользователя",
)
@limiter.limit(lambda: get_settings().rate_limit_register)
async def register(
    payload: RegisterRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> RegisterResponse:
    pw_hash = await hash_password_async(payload.password)
    email = payload.email.strip()

    async with UnitOfWork() as uow:
        # Проверка email до insert — даёт человеческий 409 вместо абстрактного IntegrityError.
        existing = await uow.users.get_by_email(email)
        if existing is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered",
            )

        user = User(email=email, password_hash=pw_hash)
        try:
            await uow.users.create(user)
        except IntegrityError as e:
            # Race: между get_by_email и create кто-то успел вставить такой же email.
            await uow.session.rollback()
            log.info("register email race", extra={"email_domain": email.split("@")[-1]})
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered",
            ) from e

        tokens, _refresh = await _issue_token_pair(
            user=user, settings=settings, request=request, uow=uow
        )
        await uow.users.update_last_login(user)
        await uow.commit()

        return RegisterResponse(user=_user_to_public(user), tokens=tokens)


@router.post(
    "/login",
    response_model=TokenPair,
    summary="Вход — пара access + refresh",
)
@limiter.limit(lambda: get_settings().rate_limit_login)
async def login(
    payload: LoginRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> TokenPair:
    # Constant-message + constant-time:
    #  - одинаковый 401 при «нет пользователя» и «неверный пароль» закрывает status leak;
    #  - выравнивание времени argon2.verify (через burn_dummy при user is None) закрывает
    #    timing-канал user enumeration. См. SECURITY review «C».
    invalid = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid email or password",
    )
    async with UnitOfWork() as uow:
        user = await uow.users.get_by_email(payload.email)
        if user is None:
            await burn_dummy_verify_async(payload.password)
            raise invalid
        if not await verify_password_async(payload.password, user.password_hash):
            raise invalid
        if user.status != "active":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account disabled",
            )

        tokens, _refresh = await _issue_token_pair(
            user=user, settings=settings, request=request, uow=uow
        )
        await uow.users.update_last_login(user)
        await uow.commit()
        return tokens


@router.post(
    "/refresh",
    response_model=TokenPair,
    summary="Ротация refresh-токена (старый revoked, новый выдан)",
)
@limiter.limit(lambda: get_settings().rate_limit_refresh)
async def refresh(
    payload: RefreshRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> TokenPair:
    invalid = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired refresh token",
    )
    token_hash = hash_refresh_token(payload.refresh_token)

    async with UnitOfWork() as uow:
        existing = await uow.refresh_tokens.get_by_token_hash(token_hash)
        if existing is None:
            raise invalid

        # Reuse-detection: если предъявлен уже revoked токен — это украденный токен.
        # Закрываем всю активную цепь, заставляем легитимного пользователя залогиниться заново.
        if existing.revoked:
            await uow.refresh_tokens.revoke_all_for_user(existing.user_id)
            await uow.commit()
            log.warning(
                "refresh token reuse detected",
                extra={"user_id": str(existing.user_id), "token_id": str(existing.id)},
            )
            raise invalid

        if existing.expires_at <= datetime.now(timezone.utc):
            raise invalid

        user = await uow.users.get_by_id(existing.user_id)
        if user is None or user.status != "active":
            raise invalid

        tokens, _new_refresh = await _issue_token_pair(
            user=user,
            settings=settings,
            request=request,
            uow=uow,
            parent=existing,
        )
        await uow.commit()
        return tokens


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Logout — отозвать предъявленный refresh-токен",
)
@limiter.limit(lambda: get_settings().rate_limit_logout)
async def logout(payload: RefreshRequest, request: Request) -> None:
    """Logout идемпотентен: повторный вызов с уже revoked токеном вернёт 204 без ошибок.

    Не требует access-токена — решение по UX: можно разлогинить даже если access протух.
    """
    token_hash = hash_refresh_token(payload.refresh_token)
    async with UnitOfWork() as uow:
        existing = await uow.refresh_tokens.get_by_token_hash(token_hash)
        if existing is not None and not existing.revoked:
            await uow.refresh_tokens.revoke(existing)
            await uow.commit()
    return None
