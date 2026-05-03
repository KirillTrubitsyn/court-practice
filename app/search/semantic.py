"""Семантический слой: Voyage клиент + cosine search по mmap-эмбеддингам."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any, Final

import httpx
import numpy as np


logger = logging.getLogger(__name__)


_VOYAGE_URL: Final = "https://api.voyageai.com/v1/embeddings"


class VoyageError(RuntimeError):
    """Любая ошибка обращения к Voyage."""


class VoyageClient:
    """Async клиент Voyage AI с retry на 429 / 5xx."""

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-3-large",
        timeout_s: float = 30.0,
        max_retries: int = 4,
    ) -> None:
        if not api_key:
            raise ValueError("voyage api_key is empty")
        self._api_key = api_key
        self._model = model
        self._max_retries = max_retries
        self._client = httpx.AsyncClient(
            timeout=timeout_s,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def embed(
        self,
        texts: list[str],
        input_type: str = "query",
    ) -> np.ndarray:
        """Вернуть L2-нормализованную матрицу эмбеддингов (len(texts), dim) float32."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        payload: dict[str, Any] = {
            "input": texts,
            "model": self._model,
            "input_type": input_type,
            "output_dtype": "float",
        }
        data = await self._post_with_retry(payload)
        rows = data.get("data") or []
        rows.sort(key=lambda x: x["index"])
        usage = data.get("usage", {})
        if usage:
            logger.info(
                "voyage_embed",
                extra={"tokens": usage.get("total_tokens"), "items": len(texts)},
            )
        vectors = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
        return _l2_normalize(vectors)

    async def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        delay = 2.0
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = await self._client.post(_VOYAGE_URL, json=payload)
            except httpx.HTTPError as exc:
                last_exc = exc
                logger.warning("voyage_network_error", extra={"attempt": attempt, "err": str(exc)})
            else:
                if resp.status_code == 200:
                    return resp.json()  # type: ignore[no-any-return]
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_exc = VoyageError(f"voyage {resp.status_code}: {resp.text[:200]}")
                    logger.warning(
                        "voyage_retryable",
                        extra={"attempt": attempt, "status": resp.status_code},
                    )
                else:
                    raise VoyageError(f"voyage {resp.status_code}: {resp.text[:500]}")

            if attempt < self._max_retries:
                await asyncio.sleep(delay)
                delay *= 2

        raise VoyageError(f"voyage retries exhausted: {last_exc}")


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


class SemanticIndex:
    """Хранит эмбеддинги корпуса (mmap). Поиск — обычное матричное умножение."""

    def __init__(self, embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings: ожидаю 2D, получил shape={embeddings.shape}")
        self._embeddings = embeddings  # (N, D), уже L2-нормализованы

    @property
    def dim(self) -> int:
        return int(self._embeddings.shape[1])

    @property
    def size(self) -> int:
        return int(self._embeddings.shape[0])

    def rank(self, query_vec: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if query_vec.shape[-1] != self._embeddings.shape[1]:
            raise ValueError(
                f"dim mismatch: query={query_vec.shape[-1]}, corpus={self._embeddings.shape[1]}"
            )
        # query_vec: (D,) или (1, D) → (N,)
        scores = self._embeddings @ query_vec.reshape(-1)
        if top_k >= scores.shape[0]:
            order = np.argsort(-scores)
        else:
            partition = np.argpartition(-scores, top_k)[:top_k]
            order = partition[np.argsort(-scores[partition])]
        return [(int(i), float(scores[i])) for i in order]

    def vector_at(self, idx: int) -> np.ndarray:
        return np.asarray(self._embeddings[idx], dtype=np.float32)


def query_hash(query: str) -> str:
    return hashlib.sha256(query.lower().strip().encode("utf-8")).hexdigest()
