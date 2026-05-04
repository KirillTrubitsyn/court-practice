"""SearchEngine — гибридный поиск с boost-факторами и дедупом по case_id.

Логика и константы взяты из reference/scripts/search.py:
- BM25 по 5 секциям с весами (title=3, vs_position=3, tags=5, full=1, fabula=0.7).
- Семантика: cosine на нормализованных эмбеддингах (отсекаем sem≤0).
- RRF top-100, w_bm25=w_sem=1.0, k=60. Финальный rrf×1000 для удобочитаемости.
- Boost: freshness (до +30%, halflife 4 года), structured_vs (×1.10), proximity (до ×1.5).
- Дедуп по case_id с накоплением alternative_channels.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final, Literal, TypedDict

import numpy as np
from rank_bm25 import BM25Okapi

from app.search.bm25 import SECTION_FIELDS, score_query as bm25_score
from app.search.fusion import reciprocal_rank_fusion
from app.search.lemmatizer import Lemmatizer
from app.search.semantic import SemanticIndex, VoyageClient, query_hash
from app.storage.redis_cache import EmbeddingCache


logger = logging.getLogger(__name__)


# Константы скоринга. Дублируют reference/scripts/search.py — не трогать без обновления тестов.
RRF_K: Final = 60
HYBRID_TOP_K: Final = 100
FRESHNESS_BOOST_MAX: Final = 0.30
FRESHNESS_HALFLIFE_YEARS: Final = 4.0
STRUCTURED_VS_BOOST: Final = 1.10
PROXIMITY_MAX_DISTANCE: Final = 3
PROXIMITY_BOOST_FACTOR: Final = 0.5  # multiplier add: 1 + factor * proximity_score
PROXIMITY_TITLE_VS_WEIGHT: Final = 1.5  # title/vs_position — внутренний вес секции в proximity
# Кейсы старше этого года считаем артефактом парсера дат (опечатки в исходных постах).
_MIN_REASONABLE_YEAR: Final = 2000


def _normalize_tag(s: str) -> str:
    """Нормализация для сравнения тегов: lower + удаление пробелов и подчёркиваний.

    Telegram-хештеги слитные (`семейныеспоры`), а пользователи естественно пишут
    `семейные споры` или `семейные_споры`. Все три должны находить друг друга.
    """
    import re

    return re.sub(r"[\s_-]+", "", s.lower())


SearchMode = Literal["hybrid", "bm25", "semantic"]


class Sections(TypedDict, total=False):
    fabula: str
    lower_courts: str
    vs_position: str
    residual: str


class Lemmas(TypedDict, total=False):
    title: list[str]
    fabula: list[str]
    lower_courts: list[str]
    vs_position: list[str]
    full: list[str]
    tags: list[str]


@dataclass(slots=True)
class Document:
    """Одна запись корпуса. Структура совпадает с reference/scripts/index.py.

    `case_id`: первый найденный канонический id (для обратной совместимости с reference).
    `case_ids`: множество всех найденных идентификаторов одного дела (номер кассации,
    арбитражный, гражданский). Дедупликация работает по пересечению этих множеств:
    два документа считаются дубликатами если у них есть хотя бы один общий case_id.
    """

    id: int  # Telegram message id
    court: str  # "СКГД" / "СКЭС" / прочее
    date: str  # сырой формат "ДД.ММ.ГГГГ" из исходника
    iso_date: str  # "YYYY-MM-DD" или ""
    year: int | None
    title: str
    case_number: str
    case_id: str  # первый канонический id (для бэкомпат с reference)
    text: str  # полный текст обзора
    hashtags: list[str]
    articles: list[str]
    source_channel: str
    sections: Sections
    lemmas: Lemmas
    case_ids: list[str] = field(default_factory=list)  # все найденные id одного дела


@dataclass(slots=True)
class IndexBundle:
    """Артефакт индексации — то, что лежит в data/index.pkl.gz.

    Структура совместима с pickle, который пишет reference/scripts/index.py:
    верхнеуровневые ключи `version, built_at, documents, bm25, case_groups, lemma_cache`.
    """

    version: int
    built_at: str
    documents: list[Document]
    bm25: dict[str, BM25Okapi]
    case_groups: dict[str, list[int]]
    lemma_cache: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class SearchHit:
    """Результат поиска. Лёгкий — для search_practice tool."""

    id: int
    title: str
    court: str
    date: str
    case_number: str
    score: float
    snippet: str
    tags: list[str]
    case_id: str = ""
    alternative_channels: list[str] = field(default_factory=list)


class SearchEngine:
    """Связывает BM25, семантику и фильтры. Создаётся ОДИН раз в lifespan сервера."""

    def __init__(
        self,
        bundle: IndexBundle,
        section_weights: dict[str, float],
        semantic: SemanticIndex | None,
        voyage: VoyageClient | None,
        cache: EmbeddingCache | None,
        rrf_k: int = RRF_K,
        bm25_weight: float = 1.0,
        semantic_weight: float = 1.0,
        hybrid_top_k: int = HYBRID_TOP_K,
    ) -> None:
        self._docs = bundle.documents
        self._bm25 = bundle.bm25
        self._case_groups = bundle.case_groups
        self._by_id: dict[int, int] = {d.id: i for i, d in enumerate(self._docs)}
        self._semantic = semantic
        self._voyage = voyage
        self._cache = cache
        self._lemmatizer = Lemmatizer(preload_cache=bundle.lemma_cache)
        self._section_weights = section_weights
        self._rrf_k = rrf_k
        self._bm25_weight = bm25_weight
        self._semantic_weight = semantic_weight
        self._hybrid_top_k = hybrid_top_k
        self._tag_counts: Counter[str] = Counter(t for d in self._docs for t in d.hashtags)
        self._meta = {
            "built_at": bundle.built_at,
            "version": bundle.version,
            "size": len(self._docs),
            "has_embeddings": semantic is not None,
        }

    # ---------- метаинформация ----------

    @property
    def size(self) -> int:
        return len(self._docs)

    @property
    def has_embeddings(self) -> bool:
        return self._semantic is not None

    def stats(self) -> dict[str, Any]:
        skgd = sum(1 for d in self._docs if "СКГД" in d.court)
        skes = sum(1 for d in self._docs if "СКЭС" in d.court)
        # Отбрасываем outliers — в отдельных Telegram-постах попадались опечатки
        # типа "10.05.1886", парсер их принимал за валидные. На реальный
        # year_range это не должно влиять (корпус 2018+).
        years = sorted({d.year for d in self._docs if d.year and d.year >= _MIN_REASONABLE_YEAR})
        with_vs = sum(1 for d in self._docs if d.sections.get("vs_position"))
        return {
            **self._meta,
            "by_court": {
                "СКГД": skgd,
                "СКЭС": skes,
                "other": len(self._docs) - skgd - skes,
            },
            "year_range": [years[0], years[-1]] if years else [],
            "with_structured_vs_position": with_vs,
            "unique_cases": len(self._case_groups),
            "unique_tags": len(self._tag_counts),
        }

    def list_tags(self, min_count: int = 5) -> list[dict[str, Any]]:
        return [
            {"tag": tag, "count": cnt}
            for tag, cnt in self._tag_counts.most_common()
            if cnt >= min_count
        ]

    def get_case(self, doc_id: int) -> Document | None:
        idx = self._by_id.get(doc_id)
        if idx is None:
            return None
        return self._docs[idx]

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
        deduplicate: bool = True,
    ) -> list[SearchHit]:
        query_lemmas = self._lemmatizer.tokenize(query) if query else []

        # Если нет ни запроса, ни фильтров — пусто (как в reference).
        if not query_lemmas and not (tag or article or court or year_from or year_to):
            return []

        # Эффективный режим: если эмбеддингов нет — fallback на чистый bm25.
        effective: SearchMode = mode
        if mode in ("hybrid", "semantic") and not self.has_embeddings:
            logger.warning("semantic_unavailable_fallback_to_bm25")
            effective = "bm25"

        eligible = self._filter(court, tag, article, year_from, year_to)
        if not eligible:
            return []

        # ---- BM25 слой ----
        bm25_scores: dict[int, float] = {}
        if query_lemmas and effective in ("bm25", "hybrid"):
            bm25_scores = bm25_score(self._bm25, query_lemmas, self._section_weights, eligible)

        # ---- Семантический слой ----
        sem_scores: dict[int, float] = {}
        if query and effective in ("semantic", "hybrid"):
            try:
                qvec = await self._embed_query(query)
            except Exception as exc:  # noqa: BLE001
                logger.warning("query_embed_failed", extra={"err": str(exc)})
                if effective == "hybrid":
                    effective = "bm25"
                else:
                    raise
            else:
                assert self._semantic is not None
                arr = self._semantic.score_all(qvec)
                for i in eligible:
                    if arr[i] > 0:
                        sem_scores[i] = float(arr[i])

        # ---- Объединение ----
        scored: list[tuple[float, int]] = []  # (score, doc_idx)

        if effective == "bm25":
            scored = [(s, i) for i, s in bm25_scores.items()]
        elif effective == "semantic":
            # Маштабируем cosine ×100 для сопоставимости с BM25 в выводе.
            scored = [(s * 100.0, i) for i, s in sem_scores.items()]
        else:  # hybrid
            scored = self._rrf_combine(bm25_scores, sem_scores)

        # ---- Boost-факторы ----
        if scored and query_lemmas:
            scored = [
                (self._apply_boosts(s, self._docs[i], query_lemmas), i) for s, i in scored
            ]

        scored.sort(key=lambda x: -x[0])

        # ---- Дедупликация по case_id ----
        return self._dedupe_and_format(scored, limit, deduplicate)

    async def find_similar(self, doc_id: int, limit: int = 5) -> list[SearchHit]:
        if not self.has_embeddings:
            raise RuntimeError(
                "Эмбеддинги корпуса не загружены. Положи embeddings.npy в DATA_DIR."
            )
        idx = self._by_id.get(doc_id)
        if idx is None:
            return []
        assert self._semantic is not None
        qvec = self._semantic.vector_at(idx)
        ranked = self._semantic.rank(qvec, top_k=limit + 1)
        out: list[SearchHit] = []
        for doc_idx, score in ranked:
            if doc_idx == idx:
                continue
            out.append(self._to_hit(self._docs[doc_idx], score * 100.0))
            if len(out) >= limit:
                break
        return out

    # ---------- внутренние методы ----------

    def _filter(
        self,
        court: str | None,
        tag: str | None,
        article: str | None,
        year_from: int | None,
        year_to: int | None,
    ) -> list[int]:
        court_norm = court.upper() if court else None
        # Fuzzy tag: Telegram-хештеги слитные ("семейныеспоры"), а пользователь
        # естественно пишет "семейные споры". Нормализуем оба к единому виду —
        # убираем все пробельные символы и приводим к lower.
        tag_normalized = _normalize_tag(tag) if tag else None
        article_lower = article.lower() if article else None
        article_nums: list[str] = []
        if article_lower:
            import re

            article_nums = re.findall(r"\d+", article_lower)

        out: list[int] = []
        for i, doc in enumerate(self._docs):
            if court_norm:
                if court_norm == "СКГД" and "СКГД" not in doc.court:
                    continue
                if court_norm == "СКЭС" and "СКЭС" not in doc.court:
                    continue
            if tag_normalized and not any(
                _normalize_tag(t) == tag_normalized for t in doc.hashtags
            ):
                continue
            if article_lower:
                text_lower = doc.text.lower()
                if article_lower not in text_lower:
                    if not (article_nums and all(n in text_lower for n in article_nums)):
                        continue
            if year_from and (not doc.year or doc.year < year_from):
                continue
            if year_to and (not doc.year or doc.year > year_to):
                continue
            out.append(i)
        return out

    def _rrf_combine(
        self,
        bm25_scores: dict[int, float],
        sem_scores: dict[int, float],
    ) -> list[tuple[float, int]]:
        bm25_top = [i for i, _ in sorted(bm25_scores.items(), key=lambda x: -x[1])[: self._hybrid_top_k]]
        sem_top = [i for i, _ in sorted(sem_scores.items(), key=lambda x: -x[1])[: self._hybrid_top_k]]
        fused = reciprocal_rank_fusion(
            rankings=[bm25_top, sem_top],
            weights=[self._bm25_weight, self._semantic_weight],
            k=self._rrf_k,
        )
        # ×1000 для удобочитаемости (как в reference/scripts/search.py).
        return [(score * 1000.0, doc_id) for doc_id, score in fused]

    def _apply_boosts(
        self,
        score: float,
        doc: Document,
        query_lemmas: list[str],
    ) -> float:
        score *= _freshness_boost(doc.year)
        if doc.sections.get("vs_position"):
            score *= STRUCTURED_VS_BOOST
        if len(query_lemmas) >= 2:
            prox = _proximity_score(doc, query_lemmas)
            if prox > 0:
                score *= 1.0 + PROXIMITY_BOOST_FACTOR * prox
        return score

    def _dedupe_and_format(
        self,
        scored: Sequence[tuple[float, int]],
        limit: int,
        deduplicate: bool,
    ) -> list[SearchHit]:
        """Свернуть дубликаты по пересечению множеств case_ids.

        Раньше дедуп шёл по точному совпадению одиночного `case_id`. Это ломалось
        когда один и тот же кейс попадал в разные telegram-каналы под разными
        форматами id (напр. 305-ЭС23-29227 vs А40-96313/2021). Теперь храним все
        найденные id и сравниваем как множества: дубликаты — у которых есть хотя
        бы один общий id.
        """
        if not deduplicate:
            return [self._to_hit(self._docs[i], s) for s, i in scored[:limit]]

        # id_to_hit: case_id → ссылка на главный hit. Все id одной группы указывают
        # на один и тот же объект, поэтому объединение групп через любой общий id
        # сразу даёт правильный лидер.
        id_to_hit: dict[str, SearchHit] = {}
        out: list[SearchHit] = []
        for score, i in scored:
            doc = self._docs[i]
            ids = self._doc_case_ids(doc)
            if not ids:
                out.append(self._to_hit(doc, score))
            else:
                # Если хотя бы один id уже видели — это дубликат
                existing = next((id_to_hit[cid] for cid in ids if cid in id_to_hit), None)
                if existing is not None:
                    if doc.source_channel and doc.source_channel not in existing.alternative_channels:
                        existing.alternative_channels.append(doc.source_channel)
                    # Расширяем индекс на новые id этой записи (могла принести unique id)
                    for cid in ids:
                        id_to_hit.setdefault(cid, existing)
                    continue
                hit = self._to_hit(doc, score)
                for cid in ids:
                    id_to_hit[cid] = hit
                out.append(hit)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _doc_case_ids(doc: Document) -> list[str]:
        """Все идентификаторы документа: case_ids + case_id (для обратной совместимости)."""
        ids = list(doc.case_ids) if doc.case_ids else []
        if doc.case_id and doc.case_id not in ids:
            ids.append(doc.case_id)
        return ids

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
            except Exception as exc:  # noqa: BLE001
                logger.warning("query_cache_set_failed", extra={"err": str(exc)})
        return vec

    def _to_hit(self, doc: Document, score: float) -> SearchHit:
        snippet = _vs_position_snippet(doc, max_length=250)
        return SearchHit(
            id=doc.id,
            title=doc.title,
            court=doc.court,
            date=doc.date,
            case_number=doc.case_number.replace("\n", " ").strip(),
            score=round(float(score), 4),
            snippet=snippet,
            tags=list(doc.hashtags),
            case_id=doc.case_id,
            alternative_channels=[],
        )


# ============================================================================
# helpers
# ============================================================================


def _freshness_boost(year: int | None) -> float:
    if not year:
        return 1.0
    years_ago = max(0, datetime.now().year - year)
    return 1.0 + FRESHNESS_BOOST_MAX * math.exp(-years_ago / FRESHNESS_HALFLIFE_YEARS)


def _proximity_score(doc: Document, query_lemmas: list[str]) -> float:
    """Совпадение пар соседних лемм запроса в окне ≤ 3 позиций по любой секции.

    Возвращает [0..1]. Совпадение в title/vs_position ценится в 1.5 раза выше fabula/full.
    """
    if len(query_lemmas) < 2:
        return 0.0
    pairs = [(query_lemmas[i], query_lemmas[i + 1]) for i in range(len(query_lemmas) - 1)]
    best = 0.0
    for section in ("title", "vs_position", "fabula", "full"):
        lemmas = doc.lemmas.get(section)  # type: ignore[call-overload]
        if not lemmas:
            continue
        positions: dict[str, list[int]] = {ql: [] for ql in query_lemmas}
        for i, lem in enumerate(lemmas):
            if lem in positions:
                positions[lem].append(i)
        matched = 0
        for a, b in pairs:
            if not positions[a] or not positions[b]:
                continue
            if any(
                0 < abs(pa - pb) <= PROXIMITY_MAX_DISTANCE
                for pa in positions[a]
                for pb in positions[b]
            ):
                matched += 1
        section_score = matched / len(pairs)
        if section in ("title", "vs_position"):
            section_score *= PROXIMITY_TITLE_VS_WEIGHT
        best = max(best, section_score)
    return min(best, 1.0)


def _vs_position_snippet(doc: Document, max_length: int = 250) -> str:
    pos = doc.sections.get("vs_position") or doc.sections.get("residual") or doc.title
    pos = pos.strip().replace("\n", " ")
    if len(pos) <= max_length:
        return pos
    cut = pos[:max_length].rfind(".")
    if cut > max_length // 2:
        return pos[: cut + 1]
    return pos[:max_length] + "..."


__all__ = [
    "Document",
    "IndexBundle",
    "SearchEngine",
    "SearchHit",
    "SearchMode",
    "Sections",
    "SECTION_FIELDS",
]
