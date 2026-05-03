"""Token endpoint. Поддерживаем grant_type = authorization_code и refresh_token."""

from __future__ import annotations

import base64
import hashlib
import logging
import secrets
import time

from starlette.requests import Request
from starlette.responses import JSONResponse

from app.oauth.jwt_utils import issue_access_token
from app.oauth.store import OAuthStores, RefreshTokenEntry


logger = logging.getLogger(__name__)


def _pkce_verify(code_verifier: str, code_challenge: str, method: str) -> bool:
    if method != "S256":
        return False
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return secrets.compare_digest(expected, code_challenge)


def make_token_handler(
    stores: OAuthStores,
    jwt_secret: str,
    access_ttl_s: int,
    refresh_ttl_s: int,
):  # type: ignore[no-untyped-def]
    async def handler(request: Request) -> JSONResponse:
        try:
            form = await request.form()
        except Exception:  # noqa: BLE001
            return JSONResponse({"error": "invalid_request"}, status_code=400)

        grant_type = (form.get("grant_type") or "").strip()

        if grant_type == "authorization_code":
            return _grant_authorization_code(form, stores, jwt_secret, access_ttl_s, refresh_ttl_s)
        if grant_type == "refresh_token":
            return _grant_refresh_token(form, stores, jwt_secret, access_ttl_s)

        return JSONResponse(
            {"error": "unsupported_grant_type"},
            status_code=400,
        )

    return handler


def _grant_authorization_code(
    form,  # type: ignore[no-untyped-def]
    stores: OAuthStores,
    jwt_secret: str,
    access_ttl_s: int,
    refresh_ttl_s: int,
) -> JSONResponse:
    code = (form.get("code") or "").strip()
    client_id = (form.get("client_id") or "").strip()
    redirect_uri = (form.get("redirect_uri") or "").strip()
    code_verifier = (form.get("code_verifier") or "").strip()

    entry = stores.codes.pop(code)  # одноразовый
    if entry is None:
        logger.warning("oauth_token_invalid_code", extra={"client_id": client_id[:8]})
        return JSONResponse({"error": "invalid_grant"}, status_code=400)

    if entry.client_id != client_id or entry.redirect_uri != redirect_uri:
        logger.warning("oauth_token_mismatch", extra={"client_id": client_id[:8]})
        return JSONResponse({"error": "invalid_grant"}, status_code=400)

    if not code_verifier or not _pkce_verify(
        code_verifier, entry.code_challenge, entry.code_challenge_method
    ):
        logger.warning("oauth_pkce_failed", extra={"client_id": client_id[:8]})
        return JSONResponse({"error": "invalid_grant"}, status_code=400)

    access_token = issue_access_token(
        secret=jwt_secret,
        user_sub=entry.user_sub,
        client_id=entry.client_id,
        scope=entry.scope,
        ttl_s=access_ttl_s,
    )
    refresh_token = secrets.token_urlsafe(32)
    stores.refresh_tokens.set(
        refresh_token,
        RefreshTokenEntry(
            client_id=entry.client_id,
            scope=entry.scope,
            user_sub=entry.user_sub,
            issued_at=int(time.time()),
        ),
        ttl_s=refresh_ttl_s,
    )
    return JSONResponse(
        {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": access_ttl_s,
            "refresh_token": refresh_token,
            "scope": entry.scope,
        }
    )


def _grant_refresh_token(
    form,  # type: ignore[no-untyped-def]
    stores: OAuthStores,
    jwt_secret: str,
    access_ttl_s: int,
) -> JSONResponse:
    refresh_token = (form.get("refresh_token") or "").strip()
    if not refresh_token:
        return JSONResponse({"error": "invalid_request"}, status_code=400)
    entry = stores.refresh_tokens.get(refresh_token)
    if entry is None:
        return JSONResponse({"error": "invalid_grant"}, status_code=400)
    access_token = issue_access_token(
        secret=jwt_secret,
        user_sub=entry.user_sub,
        client_id=entry.client_id,
        scope=entry.scope,
        ttl_s=access_ttl_s,
    )
    return JSONResponse(
        {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": access_ttl_s,
            "scope": entry.scope,
        }
    )
