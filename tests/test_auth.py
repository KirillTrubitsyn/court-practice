"""Тесты bearer-аутентификации (static + JWT)."""

from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.auth import BearerAuthMiddleware
from app.oauth.jwt_utils import issue_access_token, verify_access_token


SECRET = "x" * 48


def _no_jwt(_token: str) -> None:
    """Без JWT-проверки — в этом тесте проверяем только static path."""
    return None


@pytest.fixture
def client() -> TestClient:
    async def public(_r: Request) -> JSONResponse:
        return JSONResponse({"ok": True})

    async def protected(_r: Request) -> JSONResponse:
        return JSONResponse({"secret": "data"})

    app = Starlette(routes=[Route("/health", public), Route("/mcp", protected, methods=["POST"])])
    app.add_middleware(BearerAuthMiddleware, static_secret=SECRET, jwt_verifier=_no_jwt)
    return TestClient(app)


def test_health_is_public(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_oauth_discovery_paths_are_public(client: TestClient) -> None:
    paths = [
        "/.well-known/oauth-authorization-server",
        "/.well-known/oauth-protected-resource",
        "/.well-known/oauth-protected-resource/mcp",
        "/.well-known/openid-configuration",
        "/.well-known/anything-new",
        "/register",
        "/authorize",
        "/token",
    ]
    for path in paths:
        r = client.get(path)
        assert r.status_code != 401, f"middleware blocked {path}: got 401"


def test_protected_without_token_returns_401_with_resource_metadata(client: TestClient) -> None:
    r = client.post("/mcp")
    assert r.status_code == 401
    www_auth = r.headers.get("WWW-Authenticate", "")
    assert "Bearer" in www_auth
    # Главное — клиент должен видеть resource_metadata, чтобы пойти по OAuth flow.
    assert "resource_metadata=" in www_auth


def test_protected_wrong_static_token(client: TestClient) -> None:
    r = client.post("/mcp", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_protected_correct_static_token(client: TestClient) -> None:
    r = client.post("/mcp", headers={"Authorization": f"Bearer {SECRET}"})
    assert r.status_code == 200


def test_protected_via_jwt_verifier() -> None:
    """JWT-путь: middleware принимает токен, для которого jwt_verifier вернул payload."""

    async def protected(_r: Request) -> JSONResponse:
        return JSONResponse({"ok": True})

    def fake_verifier(token: str) -> dict | None:
        return {"sub": "shared"} if token == "valid-jwt" else None

    app = Starlette(routes=[Route("/mcp", protected, methods=["POST"])])
    app.add_middleware(BearerAuthMiddleware, static_secret=SECRET, jwt_verifier=fake_verifier)
    c = TestClient(app)

    r_ok = c.post("/mcp", headers={"Authorization": "Bearer valid-jwt"})
    assert r_ok.status_code == 200

    r_bad = c.post("/mcp", headers={"Authorization": "Bearer invalid-jwt"})
    assert r_bad.status_code == 401


def test_real_jwt_round_trip() -> None:
    """Издаём JWT нашим issue_access_token и проверяем, что middleware его принимает."""

    async def protected(_r: Request) -> JSONResponse:
        return JSONResponse({"ok": True})

    def verifier(token: str) -> dict | None:
        return verify_access_token(SECRET, token)

    app = Starlette(routes=[Route("/mcp", protected, methods=["POST"])])
    app.add_middleware(BearerAuthMiddleware, static_secret=SECRET, jwt_verifier=verifier)
    c = TestClient(app)

    token = issue_access_token(
        secret=SECRET, user_sub="shared", client_id="client-x", scope="mcp", ttl_s=60
    )
    r = c.post("/mcp", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200


def test_protected_wrong_scheme(client: TestClient) -> None:
    r = client.post("/mcp", headers={"Authorization": f"Basic {SECRET}"})
    assert r.status_code == 401
