"""Полная карточка определения по id."""

from __future__ import annotations

from dataclasses import asdict
from typing import Annotated

from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field

from app.tools._common import EngineNotReadyError, get_engine


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def get_case_details(
        case_id: Annotated[
            int,
            Field(ge=0, description="id из поля 'id' в результатах search_practice"),
        ],
        ctx: Context,
    ) -> dict:
        """Вернуть полную карточку определения: фабула, позиции судов, нормы, теги."""
        try:
            engine = get_engine(ctx)
        except EngineNotReadyError as exc:
            return {"error": "engine_not_ready", "message": str(exc)}
        case = engine.get_case(case_id)
        if case is None:
            return {"error": "not_found", "case_id": case_id}
        return asdict(case)
