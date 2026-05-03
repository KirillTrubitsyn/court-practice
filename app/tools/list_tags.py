"""Список тегов с частотами."""

from __future__ import annotations

from typing import Annotated

from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field

from app.tools._common import READ_ONLY_ANNOTATIONS, EngineNotReadyError, get_engine


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
    async def list_tags(
        ctx: Context,
        min_count: Annotated[int, Field(ge=1, le=1000)] = 5,
    ) -> list[dict] | dict:
        """Список тегов корпуса с числом определений по каждому, отсортирован по частоте."""
        try:
            engine = get_engine(ctx)
        except EngineNotReadyError as exc:
            return {"error": "engine_not_ready", "message": str(exc)}
        return engine.list_tags(min_count=min_count)
