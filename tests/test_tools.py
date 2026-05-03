"""Smoke-тесты MCP-tools: проверяем, что регистрация и dispatch работают."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from app.search.engine import SearchEngine
from app.tools._common import EngineNotReadyError, get_engine


pytestmark = pytest.mark.asyncio


@dataclass
class _State:
    engine: SearchEngine | None
    init_error: str | None = None


def _ctx(engine: SearchEngine | None) -> Any:
    return SimpleNamespace(
        request_context=SimpleNamespace(
            lifespan_context=_State(engine=engine, init_error=None if engine else "no engine")
        )
    )


async def test_get_engine_raises_when_missing() -> None:
    with pytest.raises(EngineNotReadyError):
        get_engine(_ctx(None))


async def test_search_practice_returns_dicts(engine_bm25: SearchEngine) -> None:
    from app.tools.search_practice import register

    captured: list[Any] = []

    class _StubMCP:
        def tool(self, *a: Any, **kw: Any) -> Any:  # noqa: ARG002
            def deco(fn: Any) -> Any:
                captured.append(fn)
                return fn

            return deco

    register(_StubMCP())  # type: ignore[arg-type]
    fn = captured[0]
    result = await fn(query="неустойка", ctx=_ctx(engine_bm25), mode="bm25", limit=5)
    assert isinstance(result, list)
    assert result and "snippet" in result[0]


async def test_get_case_details_returns_full_card(engine_bm25: SearchEngine) -> None:
    from app.tools.get_case_details import register

    captured: list[Any] = []

    class _StubMCP:
        def tool(self, *a: Any, **kw: Any) -> Any:  # noqa: ARG002
            def deco(fn: Any) -> Any:
                captured.append(fn)
                return fn

            return deco

    register(_StubMCP())  # type: ignore[arg-type]
    fn = captured[0]
    payload = await fn(case_id=1, ctx=_ctx(engine_bm25))
    assert payload["id"] == 1
    assert "lemmas" not in payload  # лемматизация служебка, в ответ не попадает
    assert payload["sections"]["vs_position"]


async def test_engine_not_ready_returns_error_dict(engine_bm25: SearchEngine) -> None:
    from app.tools.search_practice import register

    captured: list[Any] = []

    class _StubMCP:
        def tool(self, *a: Any, **kw: Any) -> Any:  # noqa: ARG002
            def deco(fn: Any) -> Any:
                captured.append(fn)
                return fn

            return deco

    register(_StubMCP())  # type: ignore[arg-type]
    fn = captured[0]
    result = await fn(query="x", ctx=_ctx(None), mode="bm25", limit=5)
    assert isinstance(result, dict)
    assert result["error"] == "engine_not_ready"
