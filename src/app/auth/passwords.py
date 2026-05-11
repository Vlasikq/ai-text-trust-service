"""argon2id password hashing.

Параметры по умолчанию из argon2-cffi (memory_cost=65536 KiB, time_cost=3, parallelism=4)
дают ~50–100 мс на современных CPU — достаточно для защиты от brute-force и при этом
не убивают latency на /auth/login. См. SECURITY.md для обоснования.

Sync-функции `hash_password`/`verify_password` остаются для unit-тестов; для
вызова из async-роутеров используем обёртки `*_async`, которые уводят CPU-bound
argon2 в default executor и не блокируют event loop.
"""

import asyncio

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

# Singleton hasher: при первом вызове создаёт worker thread pool; параметры по умолчанию.
_hasher = PasswordHasher()

# Подменный хеш для constant-time login. Считается один раз при импорте,
# чтобы при `email не найден` /auth/login сжигал столько же CPU, сколько на
# реальном пользователе — иначе разница ~50 мс позволяет различить
# зарегистрированные emails по времени ответа (user enumeration).
_DUMMY_HASH = _hasher.hash("dummy-password-for-constant-time-equalization")


def hash_password(password: str) -> str:
    """Вернуть argon2id-хеш пароля. Соль и параметры зашиты в строку результата."""
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Сравнить пароль с хешем. Возвращает False на mismatch / битом hash.

    Не пробрасывает исключение наружу — endpoint /auth/login не должен различать
    «нет такого пользователя» и «неверный пароль» (timing + status leak).
    """
    try:
        _hasher.verify(password_hash, password)
        return True
    except VerifyMismatchError:
        return False
    except Exception:
        # Битый формат хеша (миграция / порча в БД) — трактуем как неверный пароль.
        return False


def burn_dummy_verify(password: str) -> None:
    """Имитация verify_password против подменного хеша.

    Вызывается в /auth/login при `user is None`, чтобы выровнять время ответа
    с веткой «пользователь найден, пароль неверный». Закрывает timing-канал
    user enumeration без обращения к БД.
    """
    try:
        _hasher.verify(_DUMMY_HASH, password)
    except Exception:
        # Mismatch — ожидаемо, нас интересует только сжигание CPU.
        pass


async def hash_password_async(password: str) -> str:
    """Async-обёртка: argon2 ~50-100 мс — уводим в thread pool, не блокируем loop."""
    return await asyncio.to_thread(hash_password, password)


async def verify_password_async(password: str, password_hash: str) -> bool:
    """Async-обёртка для verify_password (см. `hash_password_async`)."""
    return await asyncio.to_thread(verify_password, password, password_hash)


async def burn_dummy_verify_async(password: str) -> None:
    """Async-обёртка для constant-time burn (см. `burn_dummy_verify`)."""
    await asyncio.to_thread(burn_dummy_verify, password)
