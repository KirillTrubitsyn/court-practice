"""End-to-end тесты OAuth 2.1 flow: discovery → register → authorize → token → /mcp."""

from __future__ import annotations

import base64
import hashlib
import secrets
from contextlib import asynccontextmanager
from urllib.parse import parse_qs, urlparse

import jwt as pyjwt
import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.auth import BearerAuthMiddleware
from app.oauth.authorization import (
    make_authorize_get_handler,
    make_authorize_post_handler,
)
from app.oauth.discovery import (
    make_authorization_server_metadata_handler,
    make_protected_resource_handler,
)
from app.oauth.jwt_utils import verify_access_token
from app.oauth.registration import make_register_handler
from app.oauth.store import OAuthStores
from app.oauth.token import make_token_handler


SECRET = "x" * 48
PASSWORD = "team-password"


def _build_app() -> Starlette:
    stores = OAuthStores.create(
        client_ttl_s=3600,
        code_ttl_s=120,
        refresh_ttl_s=3600,
    )

    async def mcp_handler(_r: Request) -> JSONResponse:
        return JSONResponse({"ok": True})

    routes = [
        Route("/.well-known/oauth-protected-resource", make_protected_resource_handler(None)),
        Route(
            "/.well-known/oauth-authorization-server",
            make_authorization_server_metadata_handler(None),
        ),
        Route("/register", make_register_handler(stores), methods=["POST"]),
        Route("/authorize", make_authorize_get_handler(stores), methods=["GET"]),
        Route(
            "/authorize",
            make_authorize_post_handler(stores, password_provider=lambda: PASSWORD),
            methods=["POST"],
        ),
        Route(
            "/token",
            make_token_handler(stores, jwt_secret=SECRET, access_ttl_s=120, refresh_ttl_s=3600),
            methods=["POST"],
        ),
        Route("/mcp", mcp_handler, methods=["POST"]),
    ]

    def verifier(token: str):  # type: ignore[no-untyped-def]
        return verify_access_token(SECRET, token)

    return Starlette(
        routes=routes,
        middleware=[
            Middleware(BearerAuthMiddleware, static_secret=SECRET, jwt_verifier=verifier),
        ],
    )


@pytest.fixture
def client() -> TestClient:
    return TestClient(_build_app())


def _pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    return verifier, challenge


# ---------- discovery ----------


def test_protected_resource_metadata(client: TestClient) -> None:
    r = client.get("/.well-known/oauth-protected-resource")
    assert r.status_code == 200
    body = r.json()
    assert body["resource"].endswith("/mcp")
    assert isinstance(body["authorization_servers"], list)
    assert "header" in body["bearer_methods_supported"]


def test_authorization_server_metadata(client: TestClient) -> None:
    r = client.get("/.well-known/oauth-authorization-server")
    assert r.status_code == 200
    body = r.json()
    assert body["authorization_endpoint"].endswith("/authorize")
    assert body["token_endpoint"].endswith("/token")
    assert body["registration_endpoint"].endswith("/register")
    assert "S256" in body["code_challenge_methods_supported"]


# ---------- registration ----------


def test_register_issues_public_client(client: TestClient) -> None:
    r = client.post(
        "/register",
        json={
            "client_name": "Claude Web",
            "redirect_uris": ["https://claude.ai/api/mcp/auth_callback"],
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["client_id"]
    assert body["token_endpoint_auth_method"] == "none"
    assert body["redirect_uris"] == ["https://claude.ai/api/mcp/auth_callback"]


def test_register_rejects_missing_redirect_uri(client: TestClient) -> None:
    r = client.post("/register", json={"client_name": "x"})
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_redirect_uri"


def test_register_rejects_non_https_uri(client: TestClient) -> None:
    r = client.post(
        "/register",
        json={"client_name": "x", "redirect_uris": ["http://evil.com/cb"]},
    )
    assert r.status_code == 400


# ---------- authorize ----------


def test_authorize_shows_login_form(client: TestClient) -> None:
    reg = client.post(
        "/register",
        json={"client_name": "x", "redirect_uris": ["https://claude.ai/cb"]},
    ).json()
    _, challenge = _pkce()
    r = client.get(
        "/authorize",
        params={
            "response_type": "code",
            "client_id": reg["client_id"],
            "redirect_uri": "https://claude.ai/cb",
            "state": "xyz",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "scope": "mcp",
        },
    )
    assert r.status_code == 200
    assert "Пароль доступа" in r.text


def test_authorize_rejects_unknown_client(client: TestClient) -> None:
    _, challenge = _pkce()
    r = client.get(
        "/authorize",
        params={
            "response_type": "code",
            "client_id": "nope",
            "redirect_uri": "https://claude.ai/cb",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
    )
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_client"


def test_authorize_redirects_with_code_on_correct_password(client: TestClient) -> None:
    reg = client.post(
        "/register",
        json={"client_name": "x", "redirect_uris": ["https://claude.ai/cb"]},
    ).json()
    _, challenge = _pkce()
    r = client.post(
        "/authorize",
        data={
            "client_id": reg["client_id"],
            "redirect_uri": "https://claude.ai/cb",
            "state": "xyz",
            "scope": "mcp",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "password": PASSWORD,
        },
        follow_redirects=False,
    )
    assert r.status_code == 302
    location = r.headers["location"]
    qs = parse_qs(urlparse(location).query)
    assert "code" in qs
    assert qs["state"] == ["xyz"]


def test_authorize_re_renders_form_on_wrong_password(client: TestClient) -> None:
    reg = client.post(
        "/register",
        json={"client_name": "x", "redirect_uris": ["https://claude.ai/cb"]},
    ).json()
    _, challenge = _pkce()
    r = client.post(
        "/authorize",
        data={
            "client_id": reg["client_id"],
            "redirect_uri": "https://claude.ai/cb",
            "scope": "mcp",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "password": "wrong",
        },
        follow_redirects=False,
    )
    assert r.status_code == 401
    assert "Неверный пароль" in r.text


# ---------- token ----------


def _full_flow_get_code(client: TestClient) -> tuple[str, str, str]:
    reg = client.post(
        "/register",
        json={"client_name": "x", "redirect_uris": ["https://claude.ai/cb"]},
    ).json()
    verifier, challenge = _pkce()
    r = client.post(
        "/authorize",
        data={
            "client_id": reg["client_id"],
            "redirect_uri": "https://claude.ai/cb",
            "scope": "mcp",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "password": PASSWORD,
        },
        follow_redirects=False,
    )
    qs = parse_qs(urlparse(r.headers["location"]).query)
    return reg["client_id"], qs["code"][0], verifier


def test_token_exchange_authorization_code(client: TestClient) -> None:
    client_id, code, verifier = _full_flow_get_code(client)
    r = client.post(
        "/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": client_id,
            "redirect_uri": "https://claude.ai/cb",
            "code_verifier": verifier,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["token_type"] == "Bearer"
    assert body["access_token"]
    assert body["refresh_token"]
    payload = pyjwt.decode(body["access_token"], SECRET, algorithms=["HS256"], options={"verify_aud": False})
    assert payload["sub"] == "shared"


def test_token_pkce_mismatch(client: TestClient) -> None:
    client_id, code, _ = _full_flow_get_code(client)
    r = client.post(
        "/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": client_id,
            "redirect_uri": "https://claude.ai/cb",
            "code_verifier": "obviously-wrong-verifier",
        },
    )
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_grant"


def test_token_code_is_one_shot(client: TestClient) -> None:
    client_id, code, verifier = _full_flow_get_code(client)
    body = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": client_id,
        "redirect_uri": "https://claude.ai/cb",
        "code_verifier": verifier,
    }
    first = client.post("/token", data=body)
    assert first.status_code == 200
    second = client.post("/token", data=body)
    assert second.status_code == 400


def test_refresh_token_grant(client: TestClient) -> None:
    client_id, code, verifier = _full_flow_get_code(client)
    first = client.post(
        "/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": client_id,
            "redirect_uri": "https://claude.ai/cb",
            "code_verifier": verifier,
        },
    ).json()
    refreshed = client.post(
        "/token",
        data={"grant_type": "refresh_token", "refresh_token": first["refresh_token"]},
    )
    assert refreshed.status_code == 200
    assert refreshed.json()["access_token"]


# ---------- end-to-end: токен реально работает на /mcp ----------


def test_issued_jwt_unlocks_mcp(client: TestClient) -> None:
    client_id, code, verifier = _full_flow_get_code(client)
    token = client.post(
        "/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": client_id,
            "redirect_uri": "https://claude.ai/cb",
            "code_verifier": verifier,
        },
    ).json()["access_token"]
    r = client.post("/mcp", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
