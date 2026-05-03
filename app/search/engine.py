"""Гибридный поисковик. Тяжёлые объекты (BM25, эмбеддинги) живут в памяти процесса."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np

from app.search.bm25 import BM25Index
from app.search.fusion import reciprocal_rank_fusion
from app.search.semantic import SemanticIndex, VoyageClient, query_hash
from app.storage.redis_cache import EmbeddingCache


logger = logging.getLogger(__name__)


SearchMode = Literal["hybrid", "lexical", "semantic"]


@dataclass(slots=True)
class Case:
    """Карточка определения ВС. Поля повторяют схему court_practice_db.json."""

    id: int
    title: str
    court: str  # СКГД | СКЭС
    date: str  # ISO YYYY-MM-DD
    case_number: str
    fabula: str
    lower_courts_position: str
    vs_position: str
    tags: list[str] = field(default_factory=list)
    articles: list[str] = field(default_factory=list)
    source_channel: str = ""
    url: str | None = None

    def search_text(self) -> str:
        # Конкатенация для одиночного эмбеддинга и BM25-документа.
        return "\n".join(
            part
            for part in (
                self.title,
                self.fabula,
                self.lower_courts_position,
                self.vs_position,
                " ".join(self.tags),
                " ".join(self.articles),
            )
            if part
        )


@dataclass(slots=True)
class IndexBundle:
    """Артефакт индексации: всё, что нужно поднять с диска при старте."""

    cases: list[Case]
    bm25: BM25Index
    tag_counts: Counter[str]
    article_counts: Counter[str]
    built_at: str  # ISO datetime
    corpus_hash: str  # хэш входного JSON, чтобы детектить рассинхрон с эмбеддингами
    voyage_model: str = ""  # модель, которой били эмбеддинги (если есть)


@dataclass(slots=True)
class SearchHit:
    id: int
    title: str
    court: str
    date: str
    case_number: str
    score: float
    snippet: str
    tags: list[str]


class SearchEngine:
    """Связывает BM25, семантику и фильтры. Создаётся ОДИН раз в lifespan."""

    def __init__(
        self,
        bundle: IndexBundle,
        semantic: SemanticIndex | None,
        voyage: VoyageClient | None,
        cache: EmbeddingCache | None,
        rrf_k: int,
        bm25_weight: float,
        semantic_weight: float,
    ) -> None:
        self._cases = bundle.cases
        self._by_id: dict[int, int] = {c.id: i for i, c in enumerate(bundle.cases)}
        self._bm25 = bundle.bm25
        self._semantic = semantic
        self._voyage = voyage
        self._cache = cache
        self._rrf_k = rrf_k
        self._bm25_weight = bm25_weight
        self._semantic_weight = semantic_weight
        self._tag_counts = bundle.tag_counts
        self._article_counts = bundle.article_counts
        self._meta = {
            "built_at": bundle.built_at,
            "voyage_model": bundle.voyage_model,
            "corpus_hash": bundle.corpus_hash,
            "size": len(bundle.cases),
            "has_embeddings": semantic is not None,
        }

    # ---------- метаданные ----------

    @property
    def size(self) -> int:
        return len(self._cases)

    @property
    def has_embeddings(self) -> bool:
        return self._semantic is not None

    def stats(self) -> dict[str, object]:
        courts = Counter(c.court for c in self._cases)
        years = Counter(c.date[:4] for c in self._cases if c.date)
        return {
            **self._meta,
            "by_court": dict(courts),
            "by_year": dict(sorted(years.items())),
            "unique_tags": len(self._tag_counts),
            "unique_articles": len(self._article_counts),
        }

    def list_tags(self, min_count: int = 5) -> list[dict[str, object]]:
        return [
            {"tag": tag, "count": cnt}
            for tag, cnt in self._tag_counts.most_common()
            if cnt >= min_count
        ]

    def get_case(self, case_id: int) -> Case | None:
        idx = self._by_id.get(case_id)
        if idx is None:
            return None
        return self._cases[idx]

    # ---------- основной поиск ----------

    async def search(
        self,
        query: str,
        mode: SearchMode = "hybrid",
        court: str | None = None,
        tag: str | None = None,
        article: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        limit: int = 10,
    ) -> list[SearchHit]:
        if not query or not query.strip():
            return []

        # Запросим больший кандидат-пул, чтобы фильтры не выкосили результат.
        pool = max(limit * 5, 50)

        # Эффективный режим: если эмбеддингов нет — fallback на lexical.
        effective_mode: SearchMode = mode
        if mode in ("hybrid", "semantic") and not self.has_embeddings:
            logger.warning("semantic_unavailable_fallback_to_lexical")
            effective_mode = "lexical"

        rankings: list[Sequence[int]] = []
        weights: list[float] = []

        if effective_mode in ("lexical", "hybrid"):
            lex_ids = self._bm25.rank(query, top_k=pool)
            if lex_ids:
                rankings.append(lex_ids)
                weights.append(self._bm25_weight)

        if effective_mode in ("semantic", "hybrid"):
            assert self._semantic is not None  # guarded above
            qvec = await self._embed_query(query)
            sem_ranked = self._semantic.rank(qvec, top_k=pool)
            sem_ids = [i for i, _ in sem_ranked]
            if sem_ids:
                rankings.append(sem_ids)
                weights.append(self._semantic_weight)

        if not rankings:
            return []

        fused = reciprocal_rank_fusion(rankings, weights=weights, k=self._rrf_k)

        # Фильтры применяем поверх слитого ранкинга — порядок сохраняется.
        out: list[SearchHit] = []
        for doc_idx, score in fused:
            case = self._cases[doc_idx]
            if not _passes_filters(case, court, tag, article, year_from, year_to):
                continue
            out.append(_to_hit(case, score))
            if len(out) >= limit:
                break
        return out

    async def find_similar(self, case_id: int, limit: int = 5) -> list[SearchHit]:
        if not self.has_embeddings:
            raise RuntimeError(
                "Эмбеддинги корпуса не построены. Запусти scripts/build_embeddings.py."
            )
        idx = self._by_id.get(case_id)
        if idx is None:
            return []
        assert self._semantic is not None
        qvec = self._semantic.vector_at(idx)
        ranked = self._semantic.rank(qvec, top_k=limit + 1)
        out: list[SearchHit] = []
        for doc_idx, score in ranked:
            if doc_idx == idx:
                continue
            out.append(_to_hit(self._cases[doc_idx], score))
            if len(out) >= limit:
                break
        return out

    # ---------- эмбеддинг запроса с кэшем ----------

    async def _embed_query(self, query: str) -> np.ndarray:
        assert self._voyage is not None and self._semantic is not None

        key = query_hash(query)
        if self._cache is not None:
            cached = await self._cache.get(key, dim=self._semantic.dim)
            if cached is not None:
                logger.info("query_cache_hit", extra={"key": key[:12]})
                return cached

        matrix = await self._voyage.embed([query], input_type="query")
        vec = matrix[0]

        if self._cache is not None:
            try:
                await self._cache.set(key, vec)
            except Exception as exc:  # graceful degradation
                logger.warning("query_cache_set_failed", extra={"err": str(exc)})

        return vec


def _to_hit(case: Case, score: float) -> SearchHit:
    snippet_src = case.vs_position or case.fabula or case.title
    snippet = snippet_src.strip().replace("\n", " ")[:250]
    return SearchHit(
        id=case.id,
        title=case.title,
        court=case.court,
        date=case.date,
        case_number=case.case_number,
        score=round(score, 6),
        snippet=snippet,
        tags=list(case.tags),
    )


def _passes_filters(
    case: Case,
    court: str | None,
    tag: str | None,
    article: str | None,
    year_from: int | None,
    year_to: int | None,
) -> bool:
    if court and case.court != court:
        return False
    if tag and tag not in case.tags:
        return False
    if article:
        if not any(article.lower() in a.lower() for a in case.articles):
            return False
    if year_from or year_to:
        case_year = _year_of(case.date)
        if case_year is None:
            return False
        if year_from and case_year < year_from:
            return False
        if year_to and case_year > year_to:
            return False
    return True


def _year_of(date_str: str) -> int | None:
    if not date_str or len(date_str) < 4:
        return None
    try:
        return int(date_str[:4])
    except ValueError:
        return None


__all__ = [
    "Case",
    "IndexBundle",
    "SearchEngine",
    "SearchHit",
    "SearchMode",
    "date",  # re-export для тайп-чека build-script
]
