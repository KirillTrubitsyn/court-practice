"""Конфиг сервера. Все параметры — через env, без хардкода.

`get_settings()` мемоизирован, потому что lifespan может дёргать его несколько раз.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # === Обязательные секреты ===
    voyage_api_key: str = Field(min_length=1)
    mcp_secret_key: str = Field(min_length=32, description="Bearer token; >=32 символов")

    # === Хранилища ===
    redis_url: str = "redis://localhost:6379/0"
    data_dir: Path = Path("./data")

    # === Voyage ===
    voyage_model: str = "voyage-3-large"
    voyage_embed_dim: int = 1024
    voyage_batch_size: int = 64
    voyage_timeout_s: float = 30.0

    # === Кэш ===
    cache_ttl_days: int = 30

    # === Веса гибридного поиска ===
    rrf_k: int = 60
    bm25_weight: float = 1.0
    semantic_weight: float = 1.0

    # === Веса секций BM25 (переопределяют DEFAULT_SECTION_WEIGHTS из reference) ===
    section_weight_title: float | None = None
    section_weight_vs_position: float | None = None
    section_weight_full: float | None = None
    section_weight_fabula: float | None = None
    section_weight_tags: float | None = None

    @property
    def section_weights_override(self) -> dict[str, float]:
        """Только заданные через env веса. Незаданные оставляют дефолт reference."""
        candidates = {
            "title": self.section_weight_title,
            "vs_position": self.section_weight_vs_position,
            "full": self.section_weight_full,
            "fabula": self.section_weight_fabula,
            "tags": self.section_weight_tags,
        }
        return {k: v for k, v in candidates.items() if v is not None}

    # === Безопасность транспорта MCP ===
    # MCP SDK 1.27 по умолчанию защищается от DNS rebinding (отвергает 421
    # на чужой Host header). На Railway за HTTPS-edge это не нужно — HTTPS
    # уже исключает атаку. Поэтому по умолчанию защита выключена.
    # Если хочется включить — выставь MCP_ENABLE_DNS_REBINDING_PROTECTION=true
    # и MCP_ALLOWED_HOSTS со списком разрешённых host через запятую.
    mcp_enable_dns_rebinding_protection: bool = False
    mcp_allowed_hosts: str = ""  # comma-separated
    mcp_allowed_origins: str = ""  # comma-separated

    @property
    def allowed_hosts_list(self) -> list[str]:
        return [h.strip() for h in self.mcp_allowed_hosts.split(",") if h.strip()]

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.mcp_allowed_origins.split(",") if o.strip()]

    # === OAuth (multi-user через shared password) ===
    # Если задан — на /authorize показывается форма с этим паролем.
    # Все коллеги вводят один и тот же пароль (как Wi-Fi). Если None — OAuth flow выключен,
    # работает только static Bearer (через MCP_SECRET_KEY) для Claude Desktop/Code.
    mcp_auth_password: str | None = None

    # Публичный URL сервиса. Если задан — используется в OAuth metadata вместо
    # вычисленного из request. Полезно за прокси, который не выставляет X-Forwarded-*.
    public_base_url: str | None = None

    # TTL access_token (JWT) и refresh_token.
    oauth_access_token_ttl_s: int = 24 * 60 * 60  # 24h
    oauth_refresh_token_ttl_s: int = 30 * 24 * 60 * 60  # 30d
    oauth_authorization_code_ttl_s: int = 120  # 2 мин — RFC 6749 рекомендует ≤ 10 мин

    # === Прочее ===
    log_level: str = "INFO"
    port: int = 8000

    @field_validator("data_dir", mode="before")
    @classmethod
    def _expand_data_dir(cls, v: object) -> Path:
        if isinstance(v, str):
            return Path(v).expanduser()
        if isinstance(v, Path):
            return v
        raise TypeError(f"data_dir: ожидаю str или Path, получил {type(v)!r}")

    @property
    def index_path(self) -> Path:
        return self.data_dir / "index.pkl.gz"

    @property
    def embeddings_path(self) -> Path:
        return self.data_dir / "embeddings.npy"

    @property
    def cache_ttl_seconds(self) -> int:
        return self.cache_ttl_days * 24 * 60 * 60


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # значения подтянутся из env
