"""Bearer-token middleware. Сравнение через secrets.compare_digest, чтобы избежать timing-атак."""

from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp


PUBLIC_PATHS: frozenset[str] = frozenset({"/health", "/healthz"})


class BearerAuthMiddleware:
    """Простейшая аутентификация. Один секрет на сервер — этого хватает для personal MCP."""

    def __init__(self, app: ASGIApp, secret: str) -> None:
        self.app = app
        self.secret = secret

    async def __call__(self, scope, receive, send):  # type: ignore[no-untyped-def]
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        if request.url.path in PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        if not self._is_authorized(request):
            response = JSONResponse(
                {"error": "unauthorized", "message": "Bearer token required"},
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="court-practice"'},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)

    def _is_authorized(self, request: Request) -> bool:
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return False
        return secrets.compare_digest(token, self.secret)


def require_bearer(handler: Callable[[Request], Awaitable[Response]], secret: str):
    """Декоратор для одиночных эндпоинтов (если понадобится)."""

    async def wrapper(request: Request) -> Response:
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(token, secret):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await handler(request)

    return wrapper
