"""BM25-слой: 5 индексов по секциям с заданными весами.

Веса (`SECTION_WEIGHTS`) подобраны эмпирически в reference/scripts/search.py:
    title × 3.0   — заголовок концентрирует суть
    vs_position × 3.0 — позиция ВС РФ — самое ценное
    full × 1.0    — общий контекст
    fabula × 0.7  — фон, важен меньше позиции
    tags × 5.0    — точное тематическое совпадение

`get_scores` для каждой секции возвращает массив длины N. Финальный скор
документа — линейная комбинация по весам.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from rank_bm25 import BM25Okapi


SECTION_FIELDS: Final[tuple[str, ...]] = ("title", "fabula", "vs_position", "full", "tags")

# Default веса. SearchEngine получает фактические из конфига и может переопределить.
DEFAULT_SECTION_WEIGHTS: Final[dict[str, float]] = {
    "title": 3.0,
    "vs_position": 3.0,
    "full": 1.0,
    "fabula": 0.7,
    "tags": 5.0,
}


def build_bm25_indexes(
    documents: list[dict[str, list[str]]],
) -> dict[str, BM25Okapi]:
    """Каждый документ — словарь leммa-списков по секциям. Возвращаем dict секция → BM25Okapi.

    rank_bm25 ругается на пустой documents, поэтому подменяем заглушкой `[""]`.
    """
    out: dict[str, BM25Okapi] = {}
    for field in SECTION_FIELDS:
        corpus = [doc.get(field) or [""] for doc in documents]
        out[field] = BM25Okapi(corpus)
    return out


def score_query(
    indexes: dict[str, BM25Okapi],
    query_lemmas: list[str],
    weights: dict[str, float],
    eligible_indices: list[int] | None = None,
) -> dict[int, float]:
    """Линейная комбинация BM25 по 5 секциям. Возвращает {doc_idx: total_score}.

    Если `eligible_indices` задан — считаем только для них (после фильтра); иначе для всех.
    Документы с total ≤ 0 отбрасываем — они не несут сигнала.
    """
    if not query_lemmas:
        return {}

    section_scores: dict[str, np.ndarray] = {
        field: indexes[field].get_scores(query_lemmas) for field in SECTION_FIELDS
    }

    target = eligible_indices if eligible_indices is not None else range(len(section_scores["title"]))
    out: dict[int, float] = {}
    for i in target:
        total = 0.0
        for field in SECTION_FIELDS:
            total += float(section_scores[field][i]) * weights.get(field, 0.0)
        if total > 0.0:
            out[i] = total
    return out
