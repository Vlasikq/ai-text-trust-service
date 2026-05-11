"""Authentication subsystem: argon2 password hashing, JWT issue/verify, FastAPI dependencies.

Контракт:
  - argon2id для паролей (стойкость к GPU, время-параметризованный)
  - access JWT (HS256, 15 мин) — подписан JWT_SECRET, кладётся в Authorization: Bearer
  - refresh: opaque random (256 бит), хранится sha256 в refresh_tokens.token_hash
  - rotation: каждый refresh отзывает старый и выдаёт новый
  - reuse-detection: попытка использовать revoked-токен → revoke всей цепи пользователя
"""

from app.auth.passwords import hash_password, verify_password
from app.auth.tokens import (
    JWTPayload,
    create_access_token,
    create_refresh_token,
    decode_access_token,
    hash_refresh_token,
)

__all__ = [
    "hash_password",
    "verify_password",
    "JWTPayload",
    "create_access_token",
    "create_refresh_token",
    "decode_access_token",
    "hash_refresh_token",
]
