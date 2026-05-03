"""Тесты bearer-аутентификации."""

from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.auth import BearerAuthMiddleware


SECRET = "x" * 48


@pytest.fixture
def client() -> TestClient:
    async def public(_r: Request) -> JSONResponse:
        return JSONResponse({"ok": True})

    async def protected(_r: Request) -> JSONResponse:
        return JSONResponse({"secret": "data"})

    app = Starlette(routes=[Route("/health", public), Route("/mcp", protected, methods=["POST"])])
    app.add_middleware(BearerAuthMiddleware, secret=SECRET)
    return TestClient(app)


def test_health_is_public(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_oauth_discovery_paths_are_public(client: TestClient) -> None:
    """Без токена эти пути не должны блокироваться middleware (404 — норма, не 401)."""
    paths = [
        "/.well-known/oauth-authorization-server",
        "/.well-known/oauth-protected-resource",
        "/.well-known/oauth-protected-resource/mcp",
        "/.well-known/openid-configuration",
        "/.well-known/anything-new",  # prefix-pass
        "/register",
    ]
    for path in paths:
        r = client.get(path)
        # Тестовое приложение не определяет эти routes — Starlette вернёт 404.
        # Главное — НЕ 401: middleware пропустил запрос дальше.
        assert r.status_code != 401, f"middleware blocked {path}: got 401"


def test_protected_without_token(client: TestClient) -> None:
    r = client.post("/mcp")
    assert r.status_code == 401
    assert "WWW-Authenticate" in r.headers


def test_protected_wrong_token(client: TestClient) -> None:
    r = client.post("/mcp", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_protected_correct_token(client: TestClient) -> None:
    r = client.post("/mcp", headers={"Authorization": f"Bearer {SECRET}"})
    assert r.status_code == 200
    assert r.json() == {"secret": "data"}


def test_protected_wrong_scheme(client: TestClient) -> None:
    r = client.post("/mcp", headers={"Authorization": f"Basic {SECRET}"})
    assert r.status_code == 401
