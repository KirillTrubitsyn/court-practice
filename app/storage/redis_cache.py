"""Async Redis-обёртка для кэша эмбеддингов запросов.

Хранение: float16 → 2 байта на измерение. Для voyage-3-large (1024D) это 2КБ на запрос.
30 дней TTL × 10к уникальных запросов ≈ 20 МБ — влезает в любой Railway Redis-план.
"""

from __future__ import annotations

import logging
from typing import Self

import numpy as np
import redis.asyncio as redis_async


logger = logging.getLogger(__name__)


_KEY_PREFIX = "court-practice:emb:"


class EmbeddingCache:
    """Тонкая обёртка вокруг redis-py asyncio с фиксированным форматом."""

    def __init__(self, client: redis_async.Redis, ttl_seconds: int) -> None:
        self._client = client
        self._ttl = ttl_seconds

    @classmethod
    async def connect(cls, url: str, ttl_seconds: int) -> Self | None:
        """Создать клиент и пингануть. None — если Redis недоступен (graceful)."""
        try:
            client = redis_async.from_url(url, decode_responses=False)
            await client.ping()
        except Exception as exc:
            logger.warning("redis_unavailable", extra={"url": url, "err": str(exc)})
            return None
        logger.info("redis_connected", extra={"url": url})
        return cls(client, ttl_seconds)

    async def close(self) -> None:
        try:
            await self._client.aclose()
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis_close_failed", extra={"err": str(exc)})

    async def get(self, key: str, dim: int) -> np.ndarray | None:
        try:
            raw: bytes | None = await self._client.get(_KEY_PREFIX + key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis_get_failed", extra={"err": str(exc)})
            return None
        if raw is None:
            return None
        if len(raw) != dim * 2:
            logger.warning("cache_dim_mismatch", extra={"got": len(raw), "want": dim * 2})
            return None
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)

    async def set(self, key: str, vec: np.ndarray) -> None:
        # Храним float16 — потеря точности на cosine ≈ 1e-3, незаметна на ранкинге.
        payload = vec.astype(np.float16).tobytes()
        try:
            await self._client.set(_KEY_PREFIX + key, payload, ex=self._ttl)
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis_set_failed", extra={"err": str(exc)})
