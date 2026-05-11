"""Unit-тесты auth-хелперов: argon2-passwords + JWT/refresh-tokens. БД не нужна."""

import time

import pytest

from app.auth.passwords import hash_password, verify_password
from app.auth.tokens import (
    TokenError,
    create_access_token,
    create_refresh_token,
    decode_access_token,
    hash_refresh_token,
)

# ── argon2 passwords ──────────────────────────────────────────


class TestPasswords:
    def test_hash_then_verify(self):
        h = hash_password("secret-password-123")
        assert h.startswith("$argon2")  # формат argon2 (id/i/d)
        assert verify_password("secret-password-123", h) is True

    def test_wrong_password(self):
        h = hash_password("correct horse battery staple")
        assert verify_password("incorrect horse battery staple", h) is False

    def test_two_hashes_differ_due_to_salt(self):
        a = hash_password("same-password")
        b = hash_password("same-password")
        assert a != b  # unique salt каждый раз
        assert verify_password("same-password", a)
        assert verify_password("same-password", b)

    def test_corrupted_hash_returns_false(self):
        # Битый hash — функция не должна падать, должна вернуть False.
        assert verify_password("anything", "this-is-not-an-argon2-hash") is False
        assert verify_password("anything", "") is False

    def test_unicode_password(self):
        pw = "пароль с пробелами и юникодом 🚀"
        h = hash_password(pw)
        assert verify_password(pw, h) is True
        assert verify_password(pw + " ", h) is False


# ── JWT access tokens ─────────────────────────────────────────


# 32+ байта — RFC 7518 рекомендует для HS256, pyjwt иначе пишет
# InsecureKeyLengthWarning. В проде ключ 64-hex (32 байта).
SECRET = "test-secret-do-not-use-in-prod-32bytes-min"
ISS = "aitrust"
AUD = "aitrust-api"


def _mk(**overrides):
    base = dict(user_id="u", role="user", secret=SECRET, issuer=ISS, audience=AUD)
    base.update(overrides)
    return create_access_token(**base)


class TestAccessJWT:
    def test_create_and_decode(self):
        token = _mk(user_id="user-123", ttl_minutes=15)
        payload = decode_access_token(token, secret=SECRET, issuer=ISS, audience=AUD)
        assert payload.user_id == "user-123"
        assert payload.role == "user"

    def test_wrong_secret_rejected(self):
        token = _mk()
        with pytest.raises(TokenError):
            decode_access_token(token, secret="other-secret", issuer=ISS, audience=AUD)

    def test_expired_token_rejected(self):
        # ttl=0 → exp <= iat, pyjwt должен отклонить (ExpiredSignatureError ⊂ InvalidTokenError).
        token = _mk(ttl_minutes=0)
        time.sleep(1.1)
        with pytest.raises(TokenError):
            decode_access_token(token, secret=SECRET, issuer=ISS, audience=AUD)

    def test_garbage_token_rejected(self):
        with pytest.raises(TokenError):
            decode_access_token("not.a.jwt", secret=SECRET, issuer=ISS, audience=AUD)
        with pytest.raises(TokenError):
            decode_access_token("", secret=SECRET, issuer=ISS, audience=AUD)

    def test_wrong_issuer_rejected(self):
        token = _mk(issuer="evil-issuer")
        with pytest.raises(TokenError):
            decode_access_token(token, secret=SECRET, issuer=ISS, audience=AUD)

    def test_wrong_audience_rejected(self):
        token = _mk(audience="some-other-api")
        with pytest.raises(TokenError):
            decode_access_token(token, secret=SECRET, issuer=ISS, audience=AUD)

    def test_algorithm_none_rejected_on_encode(self):
        """'none' алгоритм отбрасывается до подписи."""
        with pytest.raises(TokenError, match="unsupported algorithm"):
            create_access_token(
                user_id="u1", role="user", secret=SECRET,
                issuer=ISS, audience=AUD, algorithm="none",
            )

    def test_algorithm_none_rejected_on_decode(self):
        """Ослабленный config (algorithm='none') не должен пропускать токены."""
        token = _mk()
        with pytest.raises(TokenError, match="unsupported algorithm"):
            decode_access_token(
                token, secret=SECRET, issuer=ISS, audience=AUD, algorithm="none",
            )

    def test_asymmetric_algorithm_rejected(self):
        """RS256 не должен использовать HS-секрет как public key."""
        with pytest.raises(TokenError, match="unsupported algorithm"):
            create_access_token(
                user_id="u1", role="user", secret=SECRET,
                issuer=ISS, audience=AUD, algorithm="RS256",
            )


# ── Refresh tokens (opaque) ───────────────────────────────────


class TestRefreshTokens:
    def test_create_returns_unique_tokens(self):
        a = create_refresh_token()
        b = create_refresh_token()
        assert a != b
        # 32 байта в base64url ≈ 43 символа.
        assert 30 < len(a) < 100

    def test_hash_is_deterministic_64_hex(self):
        token = "fixed-token-for-test"
        h = hash_refresh_token(token)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)
        assert hash_refresh_token(token) == h

    def test_different_tokens_different_hashes(self):
        assert hash_refresh_token("a") != hash_refresh_token("b")
