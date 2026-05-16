"""Authorization endpoint.

GET /authorize  → показывает HTML-форму с полем "Пароль доступа".
POST /authorize → проверяет пароль, выпускает authorization code, редиректит на redirect_uri.

PKCE S256 обязателен. Клиенты без code_challenge отлупаются 400.
"""

from __future__ import annotations

import html
import logging
import secrets
import time
from typing import Final
from urllib.parse import urlencode

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from app.oauth.store import AuthorizationCodeEntry, OAuthStores


logger = logging.getLogger(__name__)


_LOGIN_HTML: Final = """\
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Authorize {client_name}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #0f0f10; color: #ececec; min-height: 100vh; margin: 0;
         display: flex; align-items: center; justify-content: center; }}
  .card {{ background: #18181a; border: 1px solid #2a2a2c; border-radius: 12px;
          padding: 32px; max-width: 420px; width: 100%; }}
  h1 {{ margin: 0 0 8px; font-size: 18px; font-weight: 600; }}
  p {{ color: #a0a0a0; margin: 0 0 20px; font-size: 14px; line-height: 1.5; }}
  .client {{ background: #232325; border-radius: 8px; padding: 12px; margin: 0 0 20px;
            font-size: 13px; color: #d0d0d0; }}
  label {{ display: block; font-size: 12px; color: #b8b8b8; margin: 0 0 6px; }}
  input[type=password] {{ width: 100%; padding: 10px 12px; border-radius: 8px;
                         border: 1px solid #2a2a2c; background: #0f0f10; color: #ececec;
                         font-size: 14px; box-sizing: border-box; }}
  button {{ margin-top: 16px; width: 100%; padding: 12px; border-radius: 8px;
           border: 0; background: #c97a3d; color: white; font-size: 14px;
           font-weight: 500; cursor: pointer; }}
  button:hover {{ background: #b86a30; }}
  .error {{ background: #3a1a1a; border: 1px solid #5a2a2a; color: #ffb8b8;
           padding: 10px 12px; border-radius: 8px; font-size: 13px; margin: 0 0 16px; }}
  .footer {{ margin-top: 16px; font-size: 11px; color: #707070; text-align: center; }}
</style>
</head>
<body>
<div class="card">
  <h1>Доступ к Practice MCP</h1>
  <p>Подтвердите вход для подключения клиента.</p>
  <div class="client">
    <strong>{client_name}</strong><br>
    redirect: <code>{redirect_uri_short}</code>
  </div>
  {error_html}
  <form method="post" action="/authorize">
    <label for="password">Пароль доступа</label>
    <input id="password" name="password" type="password" autocomplete="current-password" autofocus required>
    <input type="hidden" name="client_id" value="{client_id}">
    <input type="hidden" name="redirect_uri" value="{redirect_uri}">
    <input type="hidden" name="state" value="{state}">
    <input type="hidden" name="scope" value="{scope}">
    <input type="hidden" name="code_challenge" value="{code_challenge}">
    <input type="hidden" name="code_challenge_method" value="{code_challenge_method}">
    <button type="submit">Войти</button>
  </form>
  <div class="footer">court-practice MCP · OAuth 2.1</div>
</div>
</body>
</html>
"""


def _redirect_with_error(redirect_uri: str, state: str, error: str, description: str) -> Response:
    qs = urlencode({"error": error, "error_description": description, "state": state})
    return RedirectResponse(f"{redirect_uri}?{qs}", status_code=302)


def make_authorize_get_handler(stores: OAuthStores):  # type: ignore[no-untyped-def]
    async def handler(request: Request) -> Response:
        params = dict(request.query_params)
        client_id = params.get("client_id", "")
        redirect_uri = params.get("redirect_uri", "")
        state = params.get("state", "")
        scope = params.get("scope") or "mcp"
        code_challenge = params.get("code_challenge", "")
        code_challenge_method = params.get("code_challenge_method", "")
        response_type = params.get("response_type", "")

        if response_type != "code":
            return JSONResponse(
                {"error": "unsupported_response_type"},
                status_code=400,
            )
        if not client_id or not redirect_uri:
            return JSONResponse(
                {"error": "invalid_request", "error_description": "missing client_id or redirect_uri"},
                status_code=400,
            )
        client = await stores.clients.get(client_id)
        if client is None:
            return JSONResponse({"error": "invalid_client"}, status_code=400)
        if redirect_uri not in client.redirect_uris:
            return JSONResponse({"error": "invalid_redirect_uri"}, status_code=400)
        if not code_challenge or code_challenge_method != "S256":
            # После этого момента ошибки можно слать через redirect — клиент уже валиден.
            return _redirect_with_error(
                redirect_uri,
                state,
                "invalid_request",
                "PKCE S256 is required",
            )

        return HTMLResponse(
            _LOGIN_HTML.format(
                client_id=html.escape(client_id),
                client_name=html.escape(client.client_name or "MCP client"),
                redirect_uri=html.escape(redirect_uri),
                redirect_uri_short=html.escape(_short_redirect(redirect_uri)),
                state=html.escape(state),
                scope=html.escape(scope),
                code_challenge=html.escape(code_challenge),
                code_challenge_method=html.escape(code_challenge_method),
                error_html="",
            )
        )

    return handler


def make_authorize_post_handler(stores: OAuthStores, password_provider):  # type: ignore[no-untyped-def]
    """password_provider — callable() -> str | None (берётся из settings, чтобы можно было
    ротировать пароль через redeploy без рестарта процесса)."""

    async def handler(request: Request) -> Response:
        form = await request.form()
        client_id = (form.get("client_id") or "").strip()
        redirect_uri = (form.get("redirect_uri") or "").strip()
        state = (form.get("state") or "").strip()
        scope = (form.get("scope") or "mcp").strip()
        code_challenge = (form.get("code_challenge") or "").strip()
        code_challenge_method = (form.get("code_challenge_method") or "").strip()
        password = form.get("password") or ""

        client = await stores.clients.get(client_id)
        if client is None or redirect_uri not in client.redirect_uris:
            return JSONResponse({"error": "invalid_client"}, status_code=400)

        expected = password_provider()
        if expected is None:
            return JSONResponse(
                {
                    "error": "server_error",
                    "error_description": "MCP_AUTH_PASSWORD не задан в env — OAuth flow выключен",
                },
                status_code=503,
            )

        if not secrets.compare_digest(str(password), expected):
            logger.warning("oauth_login_failed", extra={"client_id": client_id[:8]})
            error_html = '<div class="error">Неверный пароль. Попробуйте ещё раз.</div>'
            return HTMLResponse(
                _LOGIN_HTML.format(
                    client_id=html.escape(client_id),
                    client_name=html.escape(client.client_name or "MCP client"),
                    redirect_uri=html.escape(redirect_uri),
                    redirect_uri_short=html.escape(_short_redirect(redirect_uri)),
                    state=html.escape(state),
                    scope=html.escape(scope),
                    code_challenge=html.escape(code_challenge),
                    code_challenge_method=html.escape(code_challenge_method),
                    error_html=error_html,
                ),
                status_code=401,
            )

        # Пароль сошёлся — выпускаем authorization code и редиректим.
        code = secrets.token_urlsafe(32)
        await stores.codes.set(
            code,
            AuthorizationCodeEntry(
                client_id=client_id,
                redirect_uri=redirect_uri,
                scope=scope,
                code_challenge=code_challenge,
                code_challenge_method=code_challenge_method or "S256",
                user_sub="shared",  # multi-user через shared password — sub один на всех
                issued_at=int(time.time()),
            ),
        )

        params = {"code": code}
        if state:
            params["state"] = state
        return RedirectResponse(f"{redirect_uri}?{urlencode(params)}", status_code=302)

    return handler


def _short_redirect(uri: str) -> str:
    if len(uri) <= 64:
        return uri
    return uri[:32] + "…" + uri[-29:]
