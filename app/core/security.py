"""Passwords and tokens (Phase 4).

  * Passwords: argon2id via argon2-cffi (library defaults). Hashes are never logged or
    returned; `verify_password` never raises. `DUMMY_HASH` is verified when the user
    does not exist so a missing account costs the same as a wrong password.
  * Tokens: HS256 JWT from PyJWT. The algorithm is pinned on decode (no `none`, no
    algorithm confusion). Claims: sub (user id), tid (tenant), iss, iat, exp, jti. The
    role is deliberately NOT trusted from the token: the API re-reads it per request
    (see app.api.dependencies), so a role change or deactivation applies at once.
"""

from __future__ import annotations

import re
import time
import uuid
from functools import lru_cache

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

ISSUER = "iia"
ALGORITHM = "HS256"
MIN_PASSWORD_CHARS = 12
MAX_PASSWORD_CHARS = 256

_hasher = PasswordHasher()          # argon2id with the library's current defaults
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class TokenError(Exception):
    """Any reason a token is unusable. Callers must not distinguish them to clients."""


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerificationError, InvalidHashError):
        return False


@lru_cache(maxsize=1)
def dummy_hash() -> str:
    """A valid argon2id hash of a random string, built once, for constant-shape logins."""
    return _hasher.hash(uuid.uuid4().hex)


def check_password_policy(password: str) -> str | None:
    """None if acceptable, else a human-readable reason."""
    if not isinstance(password, str) or len(password) < MIN_PASSWORD_CHARS:
        return f"Password must be at least {MIN_PASSWORD_CHARS} characters."
    if len(password) > MAX_PASSWORD_CHARS:
        return f"Password must be at most {MAX_PASSWORD_CHARS} characters."
    return None


def normalize_email(email: str) -> str | None:
    """Lower-cased, stripped address, or None if it is not plausibly an email."""
    if not isinstance(email, str):
        return None
    e = email.strip().lower()
    return e if len(e) <= 254 and _EMAIL_RE.match(e) else None


def create_token(*, user_id: str, tenant_id: str, secret: str, ttl_minutes: int) -> str:
    now = int(time.time())
    return jwt.encode(
        {"sub": user_id, "tid": tenant_id, "iss": ISSUER, "iat": now,
         "exp": now + ttl_minutes * 60, "jti": uuid.uuid4().hex},
        secret, algorithm=ALGORITHM)


def decode_token(token: str, secret: str) -> dict:
    try:
        claims = jwt.decode(
            token, secret, algorithms=[ALGORITHM], issuer=ISSUER,
            options={"require": ["exp", "iat", "sub", "tid", "iss", "jti"]})
    except jwt.PyJWTError as exc:
        raise TokenError(type(exc).__name__) from None
    if not isinstance(claims.get("sub"), str) or not isinstance(claims.get("tid"), str):
        raise TokenError("claims")
    return claims
