"""FastMCP сервер с Streamable HTTP транспортом.

Точка входа для uvicorn:  `uvicorn app.server:app`.

Архитектура:
- mcp = FastMCP(...) — инстанс с lifespan, внутри которого инициализируется SearchEngine.
- mcp.custom_route("/health", ...) — публичный health для Railway, без аутентификации.
- streamable_http_app() — ASGI приложение с MCP-эндпоинтом на /mcp.
- BearerAuthMiddleware повешен на это приложение; он сам пропускает /health.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from app import __version__
from app.auth import BearerAuthMiddleware
from app.config import Settings, get_settings
from app.logging_setup import configure_logging
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
    """Контейнер lifespan-state. Tools достают его через ctx.request_context.lifespan_context."""

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

    # Веса секций — env-переопределяемые; пустые наследуем из дефолта reference.
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
            },
        )
        yield state
    finally:
        logger.info("shutdown_begin")
        if cache is not None:
            await cache.close()
        await voyage.aclose()
        logger.info("shutdown_done")


# FastMCP инстанс. streamable_http_path по умолчанию "/mcp".
mcp = FastMCP(
    name="court-practice",
    instructions=(
        "Гибридный семантический поиск по обзорам определений Верховного Суда РФ "
        "(СКГД и СКЭС) за 2018–2026. Старт — search_practice, детали — get_case_details, "
        "близкие кейсы — find_similar, метаданные — stats, теги — list_tags."
    ),
    lifespan=lifespan,
)
register_all(mcp)


@mcp.custom_route("/health", methods=["GET"])
async def health(_request: Request) -> JSONResponse:
    """Публичный health endpoint для Railway healthcheck."""
    return JSONResponse({"status": "ok", "version": __version__})


def _build_app():  # noqa: ANN202
    """Собираем ASGI-приложение и навешиваем bearer-аутентификацию.

    BearerAuthMiddleware сам пропускает /health и /healthz.
    """
    settings = get_settings()
    asgi_app = mcp.streamable_http_app()
    asgi_app.add_middleware(BearerAuthMiddleware, secret=settings.mcp_secret_key)
    return asgi_app


# Готовый ASGI-инстанс для uvicorn / gunicorn.
app = _build_app()
