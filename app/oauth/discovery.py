"""OAuth discovery endpoints (RFC 8414, RFC 9728)."""

from __future__ import annotations

from typing import Final

from starlette.requests import Request
from starlette.responses import JSONResponse


_DEFAULT_SCOPE: Final = "mcp"


def base_url(request: Request, override: str | None = None) -> str:
    """Базовый URL сервиса. За прокси Railway клиент видит публичный hostname,
    но если что-то странное — позволяем явно задать через PUBLIC_BASE_URL."""
    if override:
        return override.rstrip("/")
    scheme = request.url.scheme
    host = request.headers.get("host") or request.url.netloc
    return f"{scheme}://{host}"


def make_protected_resource_handler(
    public_base_url: str | None,
):  # type: ignore[no-untyped-def]
    """`/.well-known/oauth-protected-resource` (RFC 9728).

    Объявляет, что /mcp защищён, и указывает, какие AS могут выпускать токены.
    """

    async def handler(request: Request) -> JSONResponse:
        base = base_url(request, public_base_url)
        return JSONResponse(
            {
                "resource": f"{base}/mcp",
                "authorization_servers": [base],
                "scopes_supported": [_DEFAULT_SCOPE],
                "bearer_methods_supported": ["header"],
            }
        )

    return handler


def make_authorization_server_metadata_handler(
    public_base_url: str | None,
):  # type: ignore[no-untyped-def]
    """`/.well-known/oauth-authorization-server` (RFC 8414)."""

    async def handler(request: Request) -> JSONResponse:
        base = base_url(request, public_base_url)
        return JSONResponse(
            {
                "issuer": base,
                "authorization_endpoint": f"{base}/authorize",
                "token_endpoint": f"{base}/token",
                "registration_endpoint": f"{base}/register",
                "scopes_supported": [_DEFAULT_SCOPE],
                "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code", "refresh_token"],
                "code_challenge_methods_supported": ["S256"],
                "token_endpoint_auth_methods_supported": ["none"],
            }
        )

    return handler
