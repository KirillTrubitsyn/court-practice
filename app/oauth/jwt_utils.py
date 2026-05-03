"""Выпуск и верификация JWT access_token (HS256, секрет = MCP_SECRET_KEY).

Stateless дизайн — никаких revocation list. TTL короткий (24h по умолчанию),
ротация через refresh_token. Если нужно revoke до истечения — перевыпустить
MCP_SECRET_KEY (инвалидирует все токены сразу).
"""

from __future__ import annotations

import secrets
import time
from typing import Any

import jwt


_ALGO = "HS256"
_ISSUER = "court-practice-mcp"


def issue_access_token(
    secret: str,
    user_sub: str,
    client_id: str,
    scope: str,
    ttl_s: int,
) -> str:
    now = int(time.time())
    payload: dict[str, Any] = {
        "iss": _ISSUER,
        "sub": user_sub,
        "aud": client_id,
        "scope": scope,
        "iat": now,
        "exp": now + ttl_s,
        "jti": secrets.token_urlsafe(16),
    }
    return jwt.encode(payload, secret, algorithm=_ALGO)


def verify_access_token(secret: str, token: str) -> dict[str, Any] | None:
    try:
        return jwt.decode(  # type: ignore[no-any-return]
            token,
            secret,
            algorithms=[_ALGO],
            issuer=_ISSUER,
            options={"verify_aud": False},  # audience может отличаться у разных клиентов
        )
    except jwt.PyJWTError:
        return None
