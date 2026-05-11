"""JWT access-токены + opaque refresh-токены.

Access — JWT (HS256), claims: sub (user_id), exp, iat, role. Стейтлесс — не лезем в БД.
Refresh — opaque 256-бит случайные байты; в БД хранится sha256(token), это позволяет
немедленно инвалидировать токен (revoke) и реализовать rotation + reuse-detection.
"""

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import jwt
from jwt.exceptions import InvalidTokenError

# Whitelist симметричных алгоритмов — отсекаем «alg: none» и asymmetric-confusion
# до того, как параметр дойдёт до PyJWT.
_ALLOWED_ALGORITHMS = frozenset({"HS256", "HS384", "HS512"})


class TokenError(Exception):
    """Невалидный / просроченный токен."""


def _check_algorithm(algorithm: str) -> None:
    if algorithm not in _ALLOWED_ALGORITHMS:
        raise TokenError(
            f"unsupported algorithm: {algorithm!r} "
            f"(allowed: {sorted(_ALLOWED_ALGORITHMS)})"
        )


@dataclass(frozen=True)
class JWTPayload:
    """Распарсенный access-JWT — то, что лежит в claims."""

    user_id: str
    role: str
    expires_at: datetime
    issued_at: datetime


# Access-токен (JWT).


def create_access_token(
    *,
    user_id: str,
    role: str,
    secret: str,
    issuer: str,
    audience: str,
    algorithm: str = "HS256",
    ttl_minutes: int = 15,
) -> str:
    """Подписать access-JWT с claims (sub, role, iss, aud, exp, iat)."""
    _check_algorithm(algorithm)
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "role": role,
        "iss": issuer,
        "aud": audience,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=ttl_minutes)).timestamp()),
        "type": "access",
    }
    return jwt.encode(payload, secret, algorithm=algorithm)


def decode_access_token(
    token: str,
    *,
    secret: str,
    issuer: str,
    audience: str,
    algorithm: str = "HS256",
) -> JWTPayload:
    """Проверить подпись + срок + iss/aud + тип. Бросает TokenError на любой ошибке."""
    _check_algorithm(algorithm)
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=[algorithm],
            audience=audience,
            issuer=issuer,
            options={"require": ["exp", "iat", "sub", "aud", "iss"]},
        )
    except InvalidTokenError as e:
        raise TokenError(f"invalid token: {e}") from e

    if claims.get("type") != "access":
        raise TokenError("not an access token")

    sub = claims.get("sub")
    role = claims.get("role")
    exp = claims.get("exp")
    iat = claims.get("iat")
    if not sub or not role or exp is None or iat is None:
        raise TokenError("missing required claims")

    return JWTPayload(
        user_id=str(sub),
        role=str(role),
        expires_at=datetime.fromtimestamp(exp, tz=timezone.utc),
        issued_at=datetime.fromtimestamp(iat, tz=timezone.utc),
    )


# Refresh-токен (непрозрачный).


def create_refresh_token() -> str:
    """Сгенерировать криптографически случайный refresh-токен (32 байта → 43 char base64url).

    Возвращается plain-text — выдаётся клиенту один раз. В БД хранится только sha256.
    """
    return secrets.token_urlsafe(32)


def hash_refresh_token(token: str) -> str:
    """sha256(token) в hex (64 char). Используется как ключ в refresh_tokens.token_hash.

    Hex выбран для совместимости с String(64) колонкой и удобства в логах (без бинарных байт).
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
