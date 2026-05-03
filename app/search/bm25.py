"""BM25-слой. rank_bm25 не сериализуется надёжно — храним токены, BM25 строим в памяти."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from app.search.lemmatizer import lemmatize


@dataclass
class BM25Index:
    """Контейнер BM25. Сами токены тоже храним — на случай ребилда без переиндексации."""

    tokens: list[list[str]]
    bm25: BM25Okapi

    @classmethod
    def build(cls, documents: list[str]) -> BM25Index:
        tokens = [lemmatize(doc) for doc in documents]
        # rank_bm25 ругается на пустые документы — подменяем заглушкой.
        safe = [t if t else ["__empty__"] for t in tokens]
        return cls(tokens=tokens, bm25=BM25Okapi(safe))

    def rank(self, query: str, top_k: int) -> list[int]:
        query_tokens = lemmatize(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        if top_k >= len(scores):
            order = np.argsort(-scores)
        else:
            # argpartition + сортировка только верхушки — заметно быстрее на больших корпусах.
            partition = np.argpartition(-scores, top_k)[:top_k]
            order = partition[np.argsort(-scores[partition])]
        # Отбрасываем нулевые скоры — они означают «термов не нашлось».
        return [int(i) for i in order if scores[i] > 0.0]
