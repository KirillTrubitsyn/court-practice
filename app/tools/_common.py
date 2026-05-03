"""Общие хелперы для tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.server.fastmcp import Context

from app.search.engine import SearchEngine


if TYPE_CHECKING:
    from app.server import AppState


class EngineNotReadyError(RuntimeError):
    """Индекс ещё не загружен — возвращаем понятный ответ tool-у вместо AttributeError."""


def get_engine(ctx: Context) -> SearchEngine:
    state: AppState = ctx.request_context.lifespan_context
    if state.engine is None:
        raise EngineNotReadyError(
            state.init_error
            or "SearchEngine не инициализирован. Запусти scripts/build_index.py."
        )
    return state.engine
