"""Гибридный поиск по корпусу. Возвращает «лёгкие» хиты — детали через get_case_details."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Annotated, Literal

from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field

from app.tools._common import EngineNotReadyError, get_engine


logger = logging.getLogger(__name__)


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def search_practice(
        query: Annotated[
            str,
            Field(description="Поисковый запрос на русском (естественный язык или ключевики)."),
        ],
        ctx: Context,
        mode: Annotated[
            Literal["hybrid", "lexical", "semantic"],
            Field(description="hybrid = BM25 + семантика через RRF (рекомендуется)."),
        ] = "hybrid",
        court: Annotated[
            Literal["СКГД", "СКЭС"] | None,
            Field(description="Коллегия: СКГД (гражданская) или СКЭС (экономическая)."),
        ] = None,
        tag: Annotated[str | None, Field(description="Точное совпадение тега.")] = None,
        article: Annotated[
            str | None,
            Field(description="Подстрока в норме права (например, 'ст. 333 ГК')."),
        ] = None,
        year_from: Annotated[int | None, Field(ge=2018, le=2030)] = None,
        year_to: Annotated[int | None, Field(ge=2018, le=2030)] = None,
        limit: Annotated[int, Field(ge=1, le=50)] = 10,
    ) -> list[dict] | dict:
        """Найти определения ВС РФ по запросу.

        Возвращает компактные хиты (id, title, court, date, case_number, score, snippet, tags).
        За полным текстом (фабула, позиции судов) — get_case_details(id).
        """
        try:
            engine = get_engine(ctx)
        except EngineNotReadyError as exc:
            return {"error": "engine_not_ready", "message": str(exc)}

        t0 = time.perf_counter()
        hits = await engine.search(
            query=query,
            mode=mode,
            court=court,
            tag=tag,
            article=article,
            year_from=year_from,
            year_to=year_to,
            limit=limit,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "tool_search_practice",
            extra={
                "query_len": len(query),
                "mode": mode,
                "results": len(hits),
                "latency_ms": latency_ms,
            },
        )
        return [asdict(h) for h in hits]
