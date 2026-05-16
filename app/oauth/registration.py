"""Dynamic Client Registration — RFC 7591.

claude.ai web POST-ает сюда с redirect_uris и client_name. Мы выпускаем
public client_id (без secret) и сохраняем metadata в TTL store.
"""

from __future__ import annotations

import secrets
import time

from starlette.requests import Request
from starlette.responses import JSONResponse

from app.oauth.store import OAuthStores, RegisteredClient


def make_register_handler(stores: OAuthStores):  # type: ignore[no-untyped-def]
    async def handler(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse(
                {"error": "invalid_client_metadata", "error_description": "body must be JSON"},
                status_code=400,
            )

        if not isinstance(body, dict):
            return JSONResponse(
                {"error": "invalid_client_metadata"},
                status_code=400,
            )

        redirect_uris = body.get("redirect_uris") or []
        if not isinstance(redirect_uris, list) or not redirect_uris:
            return JSONResponse(
                {
                    "error": "invalid_redirect_uri",
                    "error_description": "redirect_uris must be non-empty array",
                },
                status_code=400,
            )

        for uri in redirect_uris:
            if not isinstance(uri, str) or not uri.startswith(("https://", "http://localhost", "http://127.0.0.1")):
                return JSONResponse(
                    {
                        "error": "invalid_redirect_uri",
                        "error_description": f"redirect_uri must be https or localhost: {uri!r}",
                    },
                    status_code=400,
                )

        client_id = secrets.token_urlsafe(24)
        client_name = body.get("client_name")
        scope = body.get("scope") or "mcp"
        issued_at = int(time.time())

        await stores.clients.set(
            client_id,
            RegisteredClient(
                client_id=client_id,
                client_name=client_name,
                redirect_uris=list(redirect_uris),
                scope=scope,
                issued_at=issued_at,
            ),
        )

        return JSONResponse(
            {
                "client_id": client_id,
                "client_id_issued_at": issued_at,
                "client_name": client_name,
                "redirect_uris": redirect_uris,
                "scope": scope,
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",  # public client
            },
            status_code=201,
        )

    return handler
