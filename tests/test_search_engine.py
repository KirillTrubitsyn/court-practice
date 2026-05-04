"""Интеграционные тесты SearchEngine с multi-section BM25 + boost + dedup."""

from __future__ import annotations

import pytest

from app.search.engine import SearchEngine


pytestmark = pytest.mark.asyncio


async def test_bm25_finds_relevant(engine_bm25: SearchEngine) -> None:
    hits = await engine_bm25.search("снижение неустойки", mode="bm25", limit=5)
    assert hits, "BM25 должен находить кейсы про неустойку"
    assert hits[0].id == 1


async def test_dedup_collapses_alternative_channels(engine_bm25: SearchEngine) -> None:
    """Doc 1 и doc 4 имеют одинаковый case_id — должны схлопнуться."""
    hits = await engine_bm25.search("неустойка договор поставки", mode="bm25", limit=10)
    ids = [h.id for h in hits]
    assert 1 in ids and 4 not in ids
    primary = next(h for h in hits if h.id == 1)
    assert "ЭКОНОМКОЛЛЕГИЯ FRESH" in primary.alternative_channels


async def test_dedup_disabled_keeps_both(engine_bm25: SearchEngine) -> None:
    hits = await engine_bm25.search(
        "неустойка договор поставки", mode="bm25", limit=10, deduplicate=False
    )
    ids = {h.id for h in hits}
    assert 1 in ids and 4 in ids


async def test_dedup_collapses_via_case_ids_intersection(
    bundle, documents, lemmatizer
) -> None:
    """Multi-id дедуп: один документ с двумя case_id'ами и второй с одним из них
    должны схлопнуться, даже если их `case_id` (первый) разный."""
    from copy import deepcopy

    from app.search.bm25 import DEFAULT_SECTION_WEIGHTS
    from app.search.engine import SearchEngine

    docs = deepcopy(documents)
    # Doc 1: ВС-кассация + параллельный арбитражный номер
    docs[0].case_id = "305-ЭС24-12345"
    docs[0].case_ids = ["305-ЭС24-12345", "А40-1/2024"]
    # Doc 4: тот же арбитражный, но с другим первым case_id
    docs[3].case_id = "А40-1/2024"
    docs[3].case_ids = ["А40-1/2024"]

    engine = SearchEngine(
        bundle=bundle,
        section_weights=dict(DEFAULT_SECTION_WEIGHTS),
        semantic=None,
        voyage=None,
        cache=None,
    )
    hits = await engine.search("неустойка договор поставки", mode="bm25", limit=10)
    ids = [h.id for h in hits]
    assert 1 in ids
    assert 4 not in ids  # схлопнут через А40-1/2024 в case_ids


async def test_fuzzy_tag_matching(engine_bm25: SearchEngine) -> None:
    """tag='исковая давность' (с пробелом) должен находить хештег 'исковаяданность'/'исковая_давность'."""
    # В фикстуре hashtags=['исковая давность'] у doc 2.
    # Запрос с разделителями всё равно должен сработать.
    hits = await engine_bm25.search("", mode="bm25", tag="ИсковаяДавность", limit=10)
    ids = [h.id for h in hits]
    assert 2 in ids


async def test_stats_excludes_year_outliers(engine_bm25: SearchEngine) -> None:
    """year_range в stats не должен включать годы < 2000."""
    s = engine_bm25.stats()
    if s["year_range"]:
        assert s["year_range"][0] >= 2000


async def test_filter_by_court(engine_bm25: SearchEngine) -> None:
    hits = await engine_bm25.search("имущество супругов", mode="bm25", court="СКГД", limit=5)
    assert hits, "должны найтись кейсы СКГД"
    assert all("СКГД" in h.court for h in hits)
    assert any(h.id == 3 for h in hits)


async def test_filter_by_year_range(engine_bm25: SearchEngine) -> None:
    hits = await engine_bm25.search(
        "взыскание", mode="bm25", year_from=2024, year_to=2024, limit=10
    )
    for h in hits:
        assert h.date.endswith("2024")


async def test_filter_by_tag(engine_bm25: SearchEngine) -> None:
    hits = await engine_bm25.search("иск", mode="bm25", tag="исковая давность", limit=10)
    assert hits, "должен найтись кейс по тегу"
    assert all("исковая давность" in h.tags for h in hits)


async def test_empty_query_with_tag_works(engine_bm25: SearchEngine) -> None:
    """Чистый tag-листинг без запроса должен возвращать матчи фильтра."""
    hits = await engine_bm25.search("", mode="bm25", tag="неустойка", limit=10)
    assert hits, "пустой query + tag — это валидный фильтр-листинг"


async def test_empty_query_no_filters_returns_empty(engine_bm25: SearchEngine) -> None:
    hits = await engine_bm25.search("", mode="bm25", limit=10)
    assert hits == []


async def test_hybrid_falls_back_when_no_embeddings(engine_bm25: SearchEngine) -> None:
    hits = await engine_bm25.search("неустойка", mode="hybrid", limit=5)
    assert hits, "fallback на BM25 не должен ронять поиск"
    assert hits[0].id == 1


async def test_hybrid_with_fake_semantic(engine_hybrid: SearchEngine) -> None:
    """FakeVoyage возвращает вектор doc[0] → семантика поднимет doc 1 в топ."""
    hits = await engine_hybrid.search("любой запрос", mode="hybrid", limit=5)
    assert hits[0].id == 1


async def test_get_case_existing(engine_bm25: SearchEngine) -> None:
    doc = engine_bm25.get_case(2)
    assert doc is not None
    assert doc.title.startswith("Срок исковой давности")


async def test_get_case_missing(engine_bm25: SearchEngine) -> None:
    assert engine_bm25.get_case(9999) is None


async def test_find_similar_requires_embeddings(engine_bm25: SearchEngine) -> None:
    with pytest.raises(RuntimeError):
        await engine_bm25.find_similar(doc_id=1)


async def test_find_similar_excludes_self(engine_hybrid: SearchEngine) -> None:
    hits = await engine_hybrid.find_similar(doc_id=1, limit=2)
    assert all(h.id != 1 for h in hits)


async def test_list_tags(engine_bm25: SearchEngine) -> None:
    tags = engine_bm25.list_tags(min_count=1)
    assert tags
    assert {t["tag"] for t in tags} >= {"неустойка", "исковая давность"}


async def test_stats_structure(engine_bm25: SearchEngine) -> None:
    s = engine_bm25.stats()
    assert s["size"] == 4  # 4 документа в фикстуре
    assert "by_court" in s
    assert s["by_court"]["СКЭС"] == 2
    assert s["by_court"]["СКГД"] == 2
    assert s["unique_cases"] == 3  # doc 1 и doc 4 один case_id
