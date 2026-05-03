"""Поиск семантически близких определений по cosine similarity."""

from __future__ import annotations

from dataclasses import asdict
from typing import Annotated

from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field

from app.tools._common import READ_ONLY_ANNOTATIONS, EngineNotReadyError, get_engine


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
    async def find_similar(
        case_id: Annotated[int, Field(ge=0, description="id определения-якоря")],
        ctx: Context,
        limit: Annotated[int, Field(ge=1, le=20)] = 5,
    ) -> list[dict] | dict:
        """Найти определения, близкие к указанному по эмбеддингу (voyage-3-large)."""
        try:
            engine = get_engine(ctx)
        except EngineNotReadyError as exc:
            return {"error": "engine_not_ready", "message": str(exc)}
        try:
            hits = await engine.find_similar(doc_id=case_id, limit=limit)
        except RuntimeError as exc:
            return {"error": "embeddings_unavailable", "message": str(exc)}
        return [asdict(h) for h in hits]
