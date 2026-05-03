"""FastMCP сервер с Streamable HTTP транспортом + OAuth 2.1.

Точка входа для uvicorn: `uvicorn app.server:app`.

Архитектура:
- mcp = FastMCP(...) — инстанс с lifespan, внутри которого инициализируется SearchEngine.
- mcp.streamable_http_app() возвращает Starlette ASGI app с MCP на /mcp.
- Оборачиваем его внешним Starlette: добавляем /health + OAuth endpoints + bearer middleware.
- Для claude.ai web реализован полный OAuth 2.1 flow (DCR + PKCE S256 + JWT access_token).
- Для Claude Desktop/Code остаётся static Bearer через MCP_SECRET_KEY.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from app import __version__
from app.auth import BearerAuthMiddleware
from app.config import Settings, get_settings
from app.logging_setup import configure_logging
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
from app.search.bm25 import DEFAULT_SECTION_WEIGHTS
from app.search.engine import IndexBundle, SearchEngine
from app.search.semantic import SemanticIndex, VoyageClient
from app.storage.index_loader import (
    IndexNotFoundError,
    load_bundle,
    load_embeddings,
)
from app.storage.redis_cache import EmbeddingCache
from app.tools import register_all


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AppState:
    settings: Settings
    engine: SearchEngine | None
    voyage: VoyageClient | None
    cache: EmbeddingCache | None
    init_error: str | None = None


def _build_engine(
    bundle: IndexBundle,
    settings: Settings,
    voyage: VoyageClient | None,
    cache: EmbeddingCache | None,
) -> SearchEngine:
    embeddings = load_embeddings(settings.embeddings_path)
    semantic: SemanticIndex | None
    if embeddings is None:
        semantic = None
    elif embeddings.shape[0] != len(bundle.documents):
        logger.error(
            "embeddings_size_mismatch",
            extra={"embeddings": embeddings.shape[0], "documents": len(bundle.documents)},
        )
        semantic = None
    else:
        semantic = SemanticIndex(embeddings)

    section_weights = dict(DEFAULT_SECTION_WEIGHTS)
    section_weights.update(settings.section_weights_override)

    return SearchEngine(
        bundle=bundle,
        section_weights=section_weights,
        semantic=semantic,
        voyage=voyage if semantic is not None else None,
        cache=cache if semantic is not None else None,
        rrf_k=settings.rrf_k,
        bm25_weight=settings.bm25_weight,
        semantic_weight=settings.semantic_weight,
    )


@asynccontextmanager
async def lifespan(_server: FastMCP) -> AsyncIterator[AppState]:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info(
        "startup_begin", extra={"version": __version__, "data_dir": str(settings.data_dir)}
    )

    voyage = VoyageClient(
        api_key=settings.voyage_api_key,
        model=settings.voyage_model,
        timeout_s=settings.voyage_timeout_s,
    )
    cache = await EmbeddingCache.connect(settings.redis_url, settings.cache_ttl_seconds)

    engine: SearchEngine | None = None
    init_error: str | None = None
    try:
        bundle = load_bundle(settings.index_path)
        engine = _build_engine(bundle, settings, voyage, cache)
    except IndexNotFoundError as exc:
        init_error = str(exc)
        logger.error("index_not_found", extra={"err": init_error})
    except Exception as exc:  # noqa: BLE001
        init_error = f"index load failed: {exc}"
        logger.exception("index_load_failed")

    state = AppState(
        settings=settings,
        engine=engine,
        voyage=voyage,
        cache=cache,
        init_error=init_error,
    )
    try:
        logger.info(
            "startup_done",
            extra={
                "engine_ready": engine is not None,
                "redis": cache is not None,
                "embeddings": engine.has_embeddings if engine else False,
                "oauth_enabled": settings.mcp_auth_password is not None,
            },
        )
        yield state
    finally:
        logger.info("shutdown_begin")
        if cache is not None:
            await cache.close()
        await voyage.aclose()
        logger.info("shutdown_done")


def _transport_security(settings: Settings) -> TransportSecuritySettings | None:
    """Если защита явно включена — отдаём настройки; иначе None (защита выключена).

    На Railway за HTTPS-edge атака DNS rebinding неактуальна — браузер всё равно ходит
    через TLS, host задан в URL. Защиту имеет смысл включать, только если сервер слушает
    на голом 0.0.0.0 без HTTPS-прокси перед собой.
    """
    if not settings.mcp_enable_dns_rebinding_protection:
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=settings.allowed_hosts_list or ["*"],
        allowed_origins=settings.allowed_origins_list or ["*"],
    )


mcp = FastMCP(
    name="court-practice",
    instructions=(
        "Гибридный семантический поиск по обзорам определений Верховного Суда РФ "
        "(СКГД и СКЭС) за 2018–2026. Старт — search_practice, детали — get_case_details, "
        "близкие кейсы — find_similar, метаданные — stats, теги — list_tags."
    ),
    lifespan=lifespan,
    transport_security=_transport_security(get_settings()),
)
register_all(mcp)


async def _health(_request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "version": __version__})


def _build_app() -> Starlette:
    """Внешний Starlette с health + OAuth endpoints + Mount("/", FastMCP) + bearer middleware."""
    settings = get_settings()
    inner = mcp.streamable_http_app()

    @asynccontextmanager
    async def _lifespan(_app: Starlette) -> AsyncIterator[None]:
        async with inner.router.lifespan_context(inner):
            yield

    # OAuth state живёт всё время процесса. На рестарте теряется — клиенты переподключатся.
    stores = OAuthStores.create(
        client_ttl_s=365 * 24 * 60 * 60,  # 1 год — клиенты не должны истекать
        code_ttl_s=settings.oauth_authorization_code_ttl_s,
        refresh_ttl_s=settings.oauth_refresh_token_ttl_s,
    )

    # Handlers
    health_methods = ["GET", "POST", "HEAD"]
    discovery_methods = ["GET", "HEAD"]

    pr_handler = make_protected_resource_handler(settings.public_base_url)
    asm_handler = make_authorization_server_metadata_handler(settings.public_base_url)
    register_handler = make_register_handler(stores)
    authorize_get = make_authorize_get_handler(stores)
    authorize_post = make_authorize_post_handler(
        stores,
        password_provider=lambda: settings.mcp_auth_password,
    )
    token_handler = make_token_handler(
        stores,
        jwt_secret=settings.mcp_secret_key,
        access_ttl_s=settings.oauth_access_token_ttl_s,
        refresh_ttl_s=settings.oauth_refresh_token_ttl_s,
    )

    jwt_verifier = partial(verify_access_token, settings.mcp_secret_key)

    return Starlette(
        routes=[
            Route("/health", _health, methods=health_methods),
            Route("/healthz", _health, methods=health_methods),
            # OAuth 2.1 discovery (RFC 8414, RFC 9728).
            Route(
                "/.well-known/oauth-protected-resource",
                pr_handler,
                methods=discovery_methods,
            ),
            Route(
                "/.well-known/oauth-protected-resource/mcp",
                pr_handler,
                methods=discovery_methods,
            ),
            Route(
                "/.well-known/oauth-authorization-server",
                asm_handler,
                methods=discovery_methods,
            ),
            # OAuth flow.
            Route("/register", register_handler, methods=["POST"]),
            Route("/authorize", authorize_get, methods=["GET"]),
            Route("/authorize", authorize_post, methods=["POST"]),
            Route("/token", token_handler, methods=["POST"]),
            # MCP сам.
            Mount("/", app=inner),
        ],
        middleware=[
            Middleware(
                BearerAuthMiddleware,
                static_secret=settings.mcp_secret_key,
                jwt_verifier=jwt_verifier,
            ),
        ],
        lifespan=_lifespan,
    )


app = _build_app()
