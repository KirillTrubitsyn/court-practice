"""Общие хелперы для tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.server.fastmcp import Context
from mcp.types import ToolAnnotations

from app.search.engine import SearchEngine


if TYPE_CHECKING:
    from app.server import AppState


# Аннотации MCP-tool — это metadata для клиента (Claude.ai web/Desktop). Без них
# клиент относит инструменты в группу "Other tools" и подтверждает каждый вызов
# даже при Always allow. С readOnlyHint=True попадаем в "Read-only tools" — тогда
# Always allow реально работает.
READ_ONLY_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=True,  # сервер ходит в Voyage AI за эмбеддингами
)


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
