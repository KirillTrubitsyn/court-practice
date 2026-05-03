"""Гибридный поиск по корпусу. Возвращает «лёгкие» хиты — детали через get_case_details."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Annotated, Literal

from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field

from app.tools._common import READ_ONLY_ANNOTATIONS, EngineNotReadyError, get_engine


logger = logging.getLogger(__name__)


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
    async def search_practice(
        query: Annotated[
            str,
            Field(description="Поисковый запрос на русском (естественный язык или ключевики)."),
        ],
        ctx: Context,
        mode: Annotated[
            Literal["hybrid", "bm25", "semantic"],
            Field(description="hybrid = BM25 + семантика через RRF (рекомендуется)."),
        ] = "hybrid",
        court: Annotated[
            Literal["СКГД", "СКЭС"] | None,
            Field(description="Коллегия: СКГД (гражданская) или СКЭС (экономическая)."),
        ] = None,
        tag: Annotated[str | None, Field(description="Точное совпадение хештега (без #).")] = None,
        article: Annotated[
            str | None,
            Field(description="Подстрока в норме (например, 'ст. 333 ГК')."),
        ] = None,
        year_from: Annotated[int | None, Field(ge=2000, le=2100)] = None,
        year_to: Annotated[int | None, Field(ge=2000, le=2100)] = None,
        limit: Annotated[int, Field(ge=1, le=50)] = 10,
        deduplicate: Annotated[
            bool,
            Field(description="Сворачивать дубликаты по case_id (true по умолчанию)."),
        ] = True,
    ) -> list[dict] | dict:
        """Найти определения ВС РФ по запросу.

        Возвращает компактные хиты (id, title, court, date, case_number, score, snippet,
        tags, case_id, alternative_channels). За полным текстом — get_case_details(id).

        Подсказки:
        - mode='bm25' для точных терминов (статьи, номера дел) — экономит вызов Voyage.
        - mode='semantic' для абстрактных формулировок («снятие корпоративной вуали»).
        - Пустой query + tag/article/court — фильтр-листинг без ранкинга.
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
            deduplicate=deduplicate,
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
