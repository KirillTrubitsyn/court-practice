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
