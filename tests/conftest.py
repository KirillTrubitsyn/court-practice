"""Shared pytest fixtures."""

from __future__ import annotations

import datetime as dt
import os
from collections import Counter
from collections.abc import Iterator

import numpy as np
import pytest

# Тесты должны запускаться без реального .env. Задаём минимально необходимые env-переменные
# до импорта app.config.
os.environ.setdefault("VOYAGE_API_KEY", "test-key")
os.environ.setdefault("MCP_SECRET_KEY", "x" * 48)
os.environ.setdefault("REDIS_URL", "redis://localhost:6399/0")  # порт-заглушка
os.environ.setdefault("LOG_LEVEL", "WARNING")

from app.search.bm25 import BM25Index  # noqa: E402
from app.search.engine import Case, IndexBundle, SearchEngine  # noqa: E402
from app.search.semantic import SemanticIndex  # noqa: E402


@pytest.fixture
def sample_cases() -> list[Case]:
    return [
        Case(
            id=1,
            title="Взыскание неустойки по договору поставки",
            court="СКЭС",
            date="2023-05-10",
            case_number="А40-100/2023",
            fabula="Поставщик нарушил сроки поставки оборудования.",
            lower_courts_position="Суды снизили неустойку по ст. 333 ГК.",
            vs_position="ВС указал, что снижение неустойки требует мотивировки и доказательств несоразмерности.",
            tags=["неустойка", "договор поставки"],
            articles=["ст. 333 ГК РФ"],
        ),
        Case(
            id=2,
            title="Срок исковой давности по требованию о взыскании задолженности",
            court="СКГД",
            date="2024-01-20",
            case_number="2-50/2024",
            fabula="Истец обратился спустя 5 лет.",
            lower_courts_position="Отказано в иске.",
            vs_position="Срок исковой давности начинает течь с момента, когда лицо узнало о нарушении права.",
            tags=["исковая давность"],
            articles=["ст. 200 ГК РФ"],
        ),
        Case(
            id=3,
            title="Раздел общего имущества супругов",
            court="СКГД",
            date="2022-11-03",
            case_number="2-300/2022",
            fabula="Бывший супруг требует половину квартиры.",
            lower_courts_position="Иск удовлетворён частично.",
            vs_position="Имущество, нажитое в браке, является совместной собственностью.",
            tags=["раздел имущества", "семейное право"],
            articles=["ст. 34 СК РФ"],
        ),
    ]


@pytest.fixture
def bundle(sample_cases: list[Case]) -> IndexBundle:
    docs = [c.search_text() for c in sample_cases]
    bm25 = BM25Index.build(docs)
    return IndexBundle(
        cases=sample_cases,
        bm25=bm25,
        tag_counts=Counter(t for c in sample_cases for t in c.tags),
        article_counts=Counter(a for c in sample_cases for a in c.articles),
        built_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        corpus_hash="test",
    )


@pytest.fixture
def engine_lexical(bundle: IndexBundle) -> SearchEngine:
    return SearchEngine(
        bundle=bundle,
        semantic=None,
        voyage=None,
        cache=None,
        rrf_k=60,
        bm25_weight=1.0,
        semantic_weight=1.0,
    )


@pytest.fixture
def engine_with_fake_semantic(
    bundle: IndexBundle, sample_cases: list[Case]
) -> Iterator[SearchEngine]:
    """Подменяем SemanticIndex детерминированными векторами + фейковым Voyage."""
    rng = np.random.default_rng(42)
    matrix = rng.standard_normal((len(sample_cases), 8)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    semantic = SemanticIndex(matrix)

    class FakeVoyage:
        async def embed(self, texts, input_type="query"):  # type: ignore[no-untyped-def]
            # Эмбеддинг запроса = первая строка корпуса (значит наилучшее совпадение — case 1).
            vec = matrix[0:1].copy()
            return vec

        async def aclose(self) -> None:
            return None

    engine = SearchEngine(
        bundle=bundle,
        semantic=semantic,
        voyage=FakeVoyage(),  # type: ignore[arg-type]
        cache=None,
        rrf_k=60,
        bm25_weight=1.0,
        semantic_weight=1.0,
    )
    yield engine
