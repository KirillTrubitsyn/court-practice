"""Bearer-аутентификация.

На /mcp принимаются два типа Bearer-токенов:
1. JWT, выпущенный нашим OAuth flow (claude.ai web).
2. Static MCP_SECRET_KEY (Claude Desktop / Code, curl-тесты).

Discovery-пути и /health пропускаются без токена. На 401 отдаём заголовок
WWW-Authenticate с resource_metadata — это сигнал клиенту начать OAuth flow
(RFC 9728 § 5.1).
"""

from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp


# Точные пути, на которых не требуется bearer-токен:
# - /health, /healthz — Railway healthcheck
# - /.well-known/* — OAuth 2.1 discovery (claude.ai web дёргает их перед /mcp)
# - /register, /authorize, /token — endpoints OAuth flow (имеют свою валидацию)
PUBLIC_PATHS: frozenset[str] = frozenset(
    {
        "/health",
        "/healthz",
        "/register",
        "/authorize",
        "/token",
        "/.well-known/oauth-authorization-server",
        "/.well-known/oauth-protected-resource",
        "/.well-known/oauth-protected-resource/mcp",
        "/.well-known/openid-configuration",
    }
)

# Любой путь под этими префиксами — публичный.
PUBLIC_PREFIXES: tuple[str, ...] = ("/.well-known/",)


class BearerAuthMiddleware:
    """JWT либо static-secret. Discovery + /health пропускаются без токена.

    На 401 добавляем WWW-Authenticate с resource_metadata, чтобы MCP-клиенты
    знали, куда идти за OAuth metadata (RFC 9728).
    """

    def __init__(
        self,
        app: ASGIApp,
        static_secret: str,
        jwt_verifier: Callable[[str], dict | None],
        resource_metadata_url: str | None = None,
    ) -> None:
        self.app = app
        self._static_secret = static_secret
        self._verify_jwt = jwt_verifier
        self._resource_metadata_url = resource_metadata_url

    async def __call__(self, scope, receive, send):  # type: ignore[no-untyped-def]
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        path = request.url.path
        if path in PUBLIC_PATHS or any(path.startswith(p) for p in PUBLIC_PREFIXES):
            await self.app(scope, receive, send)
            return

        if not self._is_authorized(request):
            response = self._unauthorized(request)
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)

    def _is_authorized(self, request: Request) -> bool:
        token = self._extract_bearer(request)
        if token is None:
            return False
        # Сначала пытаемся как JWT (выпущенный нашим OAuth flow).
        if self._verify_jwt(token) is not None:
            return True
        # Fallback на static-secret — для Claude Desktop/Code и curl-тестов.
        return secrets.compare_digest(token, self._static_secret)

    @staticmethod
    def _extract_bearer(request: Request) -> str | None:
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return None
        return token

    def _unauthorized(self, request: Request) -> Response:
        # В заголовок прокладываем resource_metadata, чтобы клиент пошёл искать
        # OAuth metadata. Если URL не задан в конфиге — вычисляем из request.
        meta_url = self._resource_metadata_url
        if meta_url is None:
            base = f"{request.url.scheme}://{request.headers.get('host', request.url.netloc)}"
            meta_url = f"{base}/.well-known/oauth-protected-resource"
        www_auth = (
            f'Bearer realm="court-practice", '
            f'resource_metadata="{meta_url}"'
        )
        return JSONResponse(
            {"error": "unauthorized", "message": "Bearer token required"},
            status_code=401,
            headers={"WWW-Authenticate": www_auth},
        )


def require_bearer(handler: Callable[[Request], Awaitable[Response]], secret: str):
    """Декоратор для одиночных эндпоинтов (если понадобится)."""

    async def wrapper(request: Request) -> Response:
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(token, secret):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await handler(request)

    return wrapper
