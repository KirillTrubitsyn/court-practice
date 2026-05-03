"""Полная карточка определения по id.

Возвращает все секции (фабула, позиции нижестоящих, позиция ВС), реквизиты,
теги и нормы. Леммы и нормализованный case_id отдаются как метаинформация.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Annotated, Any

from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field

from app.tools._common import READ_ONLY_ANNOTATIONS, EngineNotReadyError, get_engine


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
    async def get_case_details(
        case_id: Annotated[
            int,
            Field(ge=0, description="id из поля 'id' в результатах search_practice"),
        ],
        ctx: Context,
    ) -> dict[str, Any]:
        """Вернуть полную карточку определения: фабула, позиции судов, нормы, теги, текст."""
        try:
            engine = get_engine(ctx)
        except EngineNotReadyError as exc:
            return {"error": "engine_not_ready", "message": str(exc)}
        doc = engine.get_case(case_id)
        if doc is None:
            return {"error": "not_found", "case_id": case_id}
        # Леммы — служебка для семантического слоя; в ответ не отдаём.
        payload = asdict(doc)
        payload.pop("lemmas", None)
        return payload
