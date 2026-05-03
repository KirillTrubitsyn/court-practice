"""Shared pytest fixtures."""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pytest

# Тесты должны запускаться без реального .env. Задаём минимально необходимые env-переменные
# до импорта app.config.
os.environ.setdefault("VOYAGE_API_KEY", "test-key")
os.environ.setdefault("MCP_SECRET_KEY", "x" * 48)
os.environ.setdefault("REDIS_URL", "redis://localhost:6399/0")
os.environ.setdefault("LOG_LEVEL", "WARNING")

from app.search.bm25 import DEFAULT_SECTION_WEIGHTS, build_bm25_indexes  # noqa: E402
from app.search.engine import Document, IndexBundle, SearchEngine  # noqa: E402
from app.search.lemmatizer import Lemmatizer  # noqa: E402
from app.search.semantic import SemanticIndex  # noqa: E402


@pytest.fixture
def lemmatizer() -> Lemmatizer:
    return Lemmatizer()


def _doc(
    *,
    id_: int,
    title: str,
    court: str,
    date: str,
    iso_date: str,
    case_number: str,
    case_id: str,
    fabula: str,
    lower_courts: str,
    vs_position: str,
    hashtags: list[str],
    articles: list[str],
    source_channel: str,
    lem: Lemmatizer,
) -> Document:
    sections = {
        "fabula": fabula,
        "lower_courts": lower_courts,
        "vs_position": vs_position,
        "residual": "",
    }
    full_text = "\n".join((title, fabula, lower_courts, vs_position))
    title_l = lem.tokenize(title)
    fabula_l = lem.tokenize(fabula)
    lower_l = lem.tokenize(lower_courts)
    vs_l = lem.tokenize(vs_position)
    tags_l: list[str] = []
    for t in hashtags:
        tags_l.extend(lem.tokenize(t))
    full_l = title_l + fabula_l + lower_l + vs_l
    return Document(
        id=id_,
        court=court,
        date=date,
        iso_date=iso_date,
        year=int(iso_date[:4]) if iso_date else None,
        title=title,
        case_number=case_number,
        case_id=case_id,
        text=full_text,
        hashtags=hashtags,
        articles=articles,
        source_channel=source_channel,
        sections=sections,  # type: ignore[arg-type]
        lemmas={  # type: ignore[arg-type]
            "title": title_l,
            "fabula": fabula_l,
            "lower_courts": lower_l,
            "vs_position": vs_l,
            "full": full_l,
            "tags": tags_l,
        },
    )


@pytest.fixture
def documents(lemmatizer: Lemmatizer) -> list[Document]:
    return [
        _doc(
            id_=1,
            title="Снижение неустойки по договору поставки",
            court="СКЭС",
            date="10.05.2024",
            iso_date="2024-05-10",
            case_number="305-ЭС24-12345",
            case_id="305-24-12345",
            fabula="Поставщик нарушил сроки поставки оборудования.",
            lower_courts="Суды снизили неустойку на основании ст. 333 ГК.",
            vs_position="Снижение неустойки по ст. 333 ГК требует мотивировки и доказательств несоразмерности.",
            hashtags=["неустойка", "договор поставки"],
            articles=["ст. 333 ГК РФ"],
            source_channel="СКЭС original",
            lem=lemmatizer,
        ),
        _doc(
            id_=2,
            title="Срок исковой давности по требованию о взыскании задолженности",
            court="СКГД",
            date="20.01.2024",
            iso_date="2024-01-20",
            case_number="5-КГ24-50",
            case_id="5-КГ24-50",
            fabula="Истец обратился спустя 5 лет после нарушения.",
            lower_courts="Отказано в иске.",
            vs_position="Срок исковой давности начинает течь с момента, когда лицо узнало о нарушении права.",
            hashtags=["исковая давность"],
            articles=["ст. 200 ГК РФ"],
            source_channel="СКГД original",
            lem=lemmatizer,
        ),
        _doc(
            id_=3,
            title="Раздел общего имущества супругов",
            court="СКГД",
            date="03.11.2022",
            iso_date="2022-11-03",
            case_number="5-КГ22-300",
            case_id="5-КГ22-300",
            fabula="Бывший супруг требует половину квартиры.",
            lower_courts="Иск удовлетворён частично.",
            vs_position="Имущество, нажитое в браке, является совместной собственностью.",
            hashtags=["раздел имущества", "семейное право"],
            articles=["ст. 34 СК РФ"],
            source_channel="СУДЕБНАЯ ПРАКТИКА",
            lem=lemmatizer,
        ),
        # Дубликат case_id у doc 1 — должен сворачиваться в alternative_channels.
        _doc(
            id_=4,
            title="Неустойка по договору поставки (дубль из другого канала)",
            court="СКЭС",
            date="10.05.2024",
            iso_date="2024-05-10",
            case_number="305-ЭС24-12345",
            case_id="305-24-12345",
            fabula="Поставщик нарушил сроки поставки оборудования.",
            lower_courts="Суды снизили неустойку.",
            vs_position="Снижение неустойки по ст. 333 ГК требует мотивировки.",
            hashtags=["неустойка"],
            articles=["ст. 333 ГК РФ"],
            source_channel="ЭКОНОМКОЛЛЕГИЯ FRESH",
            lem=lemmatizer,
        ),
    ]


@pytest.fixture
def bundle(documents: list[Document]) -> IndexBundle:
    raw_lemmas = [
        {
            "title": d.lemmas.get("title", []),
            "fabula": d.lemmas.get("fabula", []),
            "vs_position": d.lemmas.get("vs_position", []),
            "full": d.lemmas.get("full", []),
            "tags": d.lemmas.get("tags", []),
        }
        for d in documents
    ]
    bm25 = build_bm25_indexes(raw_lemmas)
    case_groups: dict[str, list[int]] = {}
    for d in documents:
        if d.case_id:
            case_groups.setdefault(d.case_id, []).append(d.id)
    return IndexBundle(
        version=1,
        built_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        documents=documents,
        bm25=bm25,
        case_groups=case_groups,
        lemma_cache={},
    )


@pytest.fixture
def engine_bm25(bundle: IndexBundle) -> SearchEngine:
    return SearchEngine(
        bundle=bundle,
        section_weights=dict(DEFAULT_SECTION_WEIGHTS),
        semantic=None,
        voyage=None,
        cache=None,
    )


@pytest.fixture
def engine_hybrid(bundle: IndexBundle, documents: list[Document]) -> SearchEngine:
    """SearchEngine с фейковой семантикой: q-вектор = эмбеддинг doc[0]."""
    rng = np.random.default_rng(42)
    matrix = rng.standard_normal((len(documents), 8)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    semantic = SemanticIndex(matrix)

    class FakeVoyage:
        async def embed(self, texts, input_type="query"):  # type: ignore[no-untyped-def]
            return matrix[0:1].copy()

        async def aclose(self) -> None:
            return None

    return SearchEngine(
        bundle=bundle,
        section_weights=dict(DEFAULT_SECTION_WEIGHTS),
        semantic=semantic,
        voyage=FakeVoyage(),  # type: ignore[arg-type]
        cache=None,
    )
