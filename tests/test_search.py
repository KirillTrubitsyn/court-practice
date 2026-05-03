"""Интеграционные тесты SearchEngine."""

from __future__ import annotations

import pytest

from app.search.engine import SearchEngine


pytestmark = pytest.mark.asyncio


async def test_lexical_search_finds_relevant(engine_lexical: SearchEngine) -> None:
    hits = await engine_lexical.search("неустойка снижение", mode="lexical", limit=5)
    assert hits, "лексический поиск должен находить кейсы"
    assert hits[0].id == 1


async def test_filter_by_court(engine_lexical: SearchEngine) -> None:
    hits = await engine_lexical.search(
        "имущество супругов",
        mode="lexical",
        court="СКГД",
        limit=5,
    )
    assert all(h.court == "СКГД" for h in hits)
    assert any(h.id == 3 for h in hits)


async def test_filter_by_year(engine_lexical: SearchEngine) -> None:
    hits = await engine_lexical.search(
        "взыскание",
        mode="lexical",
        year_from=2024,
        year_to=2024,
        limit=10,
    )
    assert all(h.date.startswith("2024") for h in hits)


async def test_filter_by_tag(engine_lexical: SearchEngine) -> None:
    hits = await engine_lexical.search(
        "иск",
        mode="lexical",
        tag="исковая давность",
        limit=10,
    )
    assert all("исковая давность" in h.tags for h in hits)


async def test_hybrid_falls_back_when_no_embeddings(engine_lexical: SearchEngine) -> None:
    # У engine_lexical нет SemanticIndex — hybrid должен молча перейти на lexical.
    hits = await engine_lexical.search("неустойка", mode="hybrid", limit=5)
    assert hits
    assert hits[0].id == 1


async def test_hybrid_with_fake_semantic(engine_with_fake_semantic: SearchEngine) -> None:
    hits = await engine_with_fake_semantic.search("любой запрос", mode="hybrid", limit=5)
    # FakeVoyage возвращает вектор case[0] → именно он должен выйти на топ через семантический слой.
    assert hits[0].id == 1


async def test_get_case_existing(engine_lexical: SearchEngine) -> None:
    case = engine_lexical.get_case(2)
    assert case is not None
    assert case.title.startswith("Срок исковой давности")


async def test_get_case_missing(engine_lexical: SearchEngine) -> None:
    assert engine_lexical.get_case(9999) is None


async def test_find_similar_requires_embeddings(engine_lexical: SearchEngine) -> None:
    with pytest.raises(RuntimeError):
        await engine_lexical.find_similar(case_id=1)


async def test_find_similar_excludes_self(engine_with_fake_semantic: SearchEngine) -> None:
    hits = await engine_with_fake_semantic.find_similar(case_id=1, limit=2)
    assert all(h.id != 1 for h in hits)


async def test_list_tags(engine_lexical: SearchEngine) -> None:
    tags = engine_lexical.list_tags(min_count=1)
    assert tags
    assert {t["tag"] for t in tags} >= {"неустойка", "исковая давность"}


async def test_stats(engine_lexical: SearchEngine) -> None:
    s = engine_lexical.stats()
    assert s["size"] == 3
    assert "by_court" in s and "by_year" in s
