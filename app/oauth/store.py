"""TTL-хранилища для DCR-клиентов, authorization codes и refresh tokens.

Два бэкенда с одинаковым async-интерфейсом:
  • TtlStore      — in-memory, для тестов и локального запуска без Redis;
  • RedisTtlStore — Redis-backed, переживает рестарты процесса.

Зачем Redis: на Railway процесс перезапускается при каждом деплое (и при
платформенном обслуживании). In-memory state при этом обнуляется → claude.ai
теряет refresh token, не может его обменять и вынужден гнать полный OAuth-flow
заново, показывая пользователю форму с паролем. RedisTtlStore кладёт state в
Redis (тот же инстанс, что и кэш эмбеддингов), поэтому коннектор остаётся
подключённым между рестартами.

RedisTtlStore при сбое Redis прозрачно падает на in-memory fallback — тогда
поведение деградирует ровно до прежнего (state теряется на рестарте), но
OAuth-flow продолжает работать.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Generic, Protocol, TypeVar

import redis.asyncio as redis_async


logger = logging.getLogger(__name__)

T = TypeVar("T")


class AsyncKVStore(Protocol[T]):
    """Async-интерфейс TTL-хранилища, общий для in-memory и Redis-бэкендов."""

    async def set(self, key: str, value: T, ttl_s: int | None = None) -> None: ...

    async def get(self, key: str) -> T | None: ...

    async def pop(self, key: str) -> T | None: ...


@dataclass(slots=True)
class _Entry(Generic[T]):
    value: T
    expires_at: float


class TtlStore(Generic[T]):
    """In-memory потокобезопасный dict с TTL. Fallback, когда Redis недоступен."""

    def __init__(self, default_ttl_s: int) -> None:
        self._default_ttl = default_ttl_s
        self._items: dict[str, _Entry[T]] = {}
        self._lock = threading.Lock()

    async def set(self, key: str, value: T, ttl_s: int | None = None) -> None:
        ttl = ttl_s if ttl_s is not None else self._default_ttl
        with self._lock:
            self._items[key] = _Entry(value=value, expires_at=time.time() + ttl)

    async def get(self, key: str) -> T | None:
        now = time.time()
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                return None
            if entry.expires_at < now:
                self._items.pop(key, None)
                return None
            return entry.value

    async def pop(self, key: str) -> T | None:
        with self._lock:
            entry = self._items.pop(key, None)
        if entry is None or entry.expires_at < time.time():
            return None
        return entry.value

    def __len__(self) -> int:
        return len(self._items)


class RedisTtlStore(Generic[T]):
    """Redis-backed TTL-хранилище. State переживает рестарт процесса.

    Значения сериализуются в JSON. При любом сбое Redis операция прозрачно
    уходит в in-memory fallback — OAuth-flow не ломается, лишь теряет
    устойчивость к рестартам (как было до перехода на Redis).
    """

    def __init__(
        self,
        client: redis_async.Redis[Any],
        key_prefix: str,
        default_ttl_s: int,
        dumps: Callable[[T], dict[str, Any]],
        loads: Callable[[dict[str, Any]], T],
    ) -> None:
        self._client = client
        self._prefix = key_prefix
        self._default_ttl = default_ttl_s
        self._dumps = dumps
        self._loads = loads
        self._fallback: TtlStore[T] = TtlStore(default_ttl_s)

    async def set(self, key: str, value: T, ttl_s: int | None = None) -> None:
        ttl = ttl_s if ttl_s is not None else self._default_ttl
        payload = json.dumps(self._dumps(value))
        try:
            await self._client.set(self._prefix + key, payload, ex=ttl)
        except Exception as exc:  # noqa: BLE001
            logger.warning("oauth_redis_set_failed", extra={"err": str(exc)})
            await self._fallback.set(key, value, ttl_s)

    async def get(self, key: str) -> T | None:
        try:
            raw = await self._client.get(self._prefix + key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("oauth_redis_get_failed", extra={"err": str(exc)})
            return await self._fallback.get(key)
        if raw is None:
            return None
        return self._decode(raw)

    async def pop(self, key: str) -> T | None:
        try:
            raw = await self._client.getdel(self._prefix + key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("oauth_redis_pop_failed", extra={"err": str(exc)})
            return await self._fallback.pop(key)
        if raw is None:
            return None
        return self._decode(raw)

    def _decode(self, raw: str | bytes) -> T | None:
        try:
            return self._loads(json.loads(raw))
        except Exception as exc:  # noqa: BLE001
            logger.warning("oauth_redis_decode_failed", extra={"err": str(exc)})
            return None


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


_REDIS_KEY_PREFIX = "court-practice:oauth:"


@dataclass(slots=True)
class OAuthStores:
    """Контейнер с тремя ttl-хранилищами OAuth-состояния."""

    clients: AsyncKVStore[RegisteredClient]
    codes: AsyncKVStore[AuthorizationCodeEntry]
    refresh_tokens: AsyncKVStore[RefreshTokenEntry]
    _redis: redis_async.Redis[Any] | None = None

    @classmethod
    def create(
        cls,
        client_ttl_s: int,
        code_ttl_s: int,
        refresh_ttl_s: int,
    ) -> OAuthStores:
        """In-memory бэкенд. State теряется на рестарте — для тестов и dev."""
        return cls(
            clients=TtlStore[RegisteredClient](default_ttl_s=client_ttl_s),
            codes=TtlStore[AuthorizationCodeEntry](default_ttl_s=code_ttl_s),
            refresh_tokens=TtlStore[RefreshTokenEntry](default_ttl_s=refresh_ttl_s),
        )

    @classmethod
    def create_redis(
        cls,
        redis_url: str,
        client_ttl_s: int,
        code_ttl_s: int,
        refresh_ttl_s: int,
    ) -> OAuthStores:
        """Redis-backed бэкенд. State переживает рестарт процесса.

        Клиент создаётся лениво (redis-py подключается при первой команде),
        поэтому вызов синхронный и не требует доступности Redis в момент старта.
        """
        client = redis_async.from_url(redis_url, decode_responses=True)
        return cls(
            clients=RedisTtlStore[RegisteredClient](
                client,
                _REDIS_KEY_PREFIX + "client:",
                client_ttl_s,
                dataclasses.asdict,
                lambda d: RegisteredClient(**d),
            ),
            codes=RedisTtlStore[AuthorizationCodeEntry](
                client,
                _REDIS_KEY_PREFIX + "code:",
                code_ttl_s,
                dataclasses.asdict,
                lambda d: AuthorizationCodeEntry(**d),
            ),
            refresh_tokens=RedisTtlStore[RefreshTokenEntry](
                client,
                _REDIS_KEY_PREFIX + "refresh:",
                refresh_ttl_s,
                dataclasses.asdict,
                lambda d: RefreshTokenEntry(**d),
            ),
            _redis=client,
        )

    async def aclose(self) -> None:
        if self._redis is not None:
            try:
                await self._redis.aclose()
            except Exception as exc:  # noqa: BLE001
                logger.warning("oauth_redis_close_failed", extra={"err": str(exc)})
