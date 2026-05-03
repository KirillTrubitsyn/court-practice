"""Метаданные корпуса: размер, разбивка по коллегиям и годам, статус эмбеддингов."""

from __future__ import annotations

from mcp.server.fastmcp import Context, FastMCP

from app.tools._common import READ_ONLY_ANNOTATIONS, EngineNotReadyError, get_engine


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
    async def stats(ctx: Context) -> dict:
        """Метаданные базы судебной практики (когда индексировано, сколько кейсов, разбивка)."""
        try:
            engine = get_engine(ctx)
        except EngineNotReadyError as exc:
            return {"error": "engine_not_ready", "message": str(exc)}
        return engine.stats()
