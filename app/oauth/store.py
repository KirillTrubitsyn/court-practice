"""In-memory TTL хранилища для DCR-клиентов, authorization codes и refresh tokens.

Зачем in-memory: на Railway Hobby у нас одна replica и ребут редкий. После рестарта
claude.ai web просто перерегистрируется (у нас это автоматический flow).

Если в будущем перейдём на multi-replica — переедем на Redis (тот же EmbeddingCache).
Интерфейс намеренно совместим.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass(slots=True)
class _Entry(Generic[T]):
    value: T
    expires_at: float


class TtlStore(Generic[T]):
    """Простой потокобезопасный dict с TTL. Для small-footprint deployments достаточно."""

    def __init__(self, default_ttl_s: int) -> None:
        self._default_ttl = default_ttl_s
        self._items: dict[str, _Entry[T]] = {}
        self._lock = threading.Lock()

    def set(self, key: str, value: T, ttl_s: int | None = None) -> None:
        ttl = ttl_s if ttl_s is not None else self._default_ttl
        with self._lock:
            self._items[key] = _Entry(value=value, expires_at=time.time() + ttl)

    def get(self, key: str) -> T | None:
        now = time.time()
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                return None
            if entry.expires_at < now:
                self._items.pop(key, None)
                return None
            return entry.value

    def pop(self, key: str) -> T | None:
        with self._lock:
            entry = self._items.pop(key, None)
        if entry is None or entry.expires_at < time.time():
            return None
        return entry.value

    def cleanup(self) -> int:
        now = time.time()
        removed = 0
        with self._lock:
            for key in list(self._items.keys()):
                if self._items[key].expires_at < now:
                    self._items.pop(key, None)
                    removed += 1
        return removed

    def __len__(self) -> int:
        return len(self._items)


# ============================================================================
# Типизированные структуры
# ============================================================================


@dataclass(slots=True)
class RegisteredClient:
    client_id: str
    client_name: str | None
    redirect_uris: list[str]
    scope: str
    issued_at: int


@dataclass(slots=True)
class AuthorizationCodeEntry:
    client_id: str
    redirect_uri: str
    scope: str
    code_challenge: str
    code_challenge_method: str  # "S256" only — мы plain не разрешаем
    user_sub: str
    issued_at: int


@dataclass(slots=True)
class RefreshTokenEntry:
    client_id: str
    scope: str
    user_sub: str
    issued_at: int


@dataclass(slots=True)
class OAuthStores:
    """Контейнер с тремя ttl-хранилищами OAuth-состояния."""

    clients: TtlStore[RegisteredClient]
    codes: TtlStore[AuthorizationCodeEntry]
    refresh_tokens: TtlStore[RefreshTokenEntry]

    @classmethod
    def create(
        cls,
        client_ttl_s: int,
        code_ttl_s: int,
        refresh_ttl_s: int,
    ) -> OAuthStores:
        return cls(
            clients=TtlStore[RegisteredClient](default_ttl_s=client_ttl_s),
            codes=TtlStore[AuthorizationCodeEntry](default_ttl_s=code_ttl_s),
            refresh_tokens=TtlStore[RefreshTokenEntry](default_ttl_s=refresh_ttl_s),
        )
