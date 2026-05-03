"""Достроить только эмбеддинги поверх уже существующего index.pkl.gz.

Полезно если:
- сначала проиндексировал без --embeddings (чтобы отладить пайплайн без расхода Voyage),
- хочешь перебилдить эмбеддинги под другой моделью без полной переиндексации BM25.

Запуск:
    python -m scripts.build_embeddings --out-dir data
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from tqdm import tqdm

from app.config import get_settings
from app.search.semantic import VoyageClient
from app.storage.index_loader import load_bundle, save_embeddings


logger = logging.getLogger("build_embeddings")


def _truncate(text: str, max_chars: int = 30_000) -> str:
    return text if len(text) <= max_chars else text[:max_chars]


@asynccontextmanager
async def _voyage_session(settings):  # type: ignore[no-untyped-def]
    client = VoyageClient(
        api_key=settings.voyage_api_key,
        model=settings.voyage_model,
        timeout_s=settings.voyage_timeout_s,
    )
    try:
        yield client
    finally:
        await client.aclose()


async def amain() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    settings = get_settings()
    bundle = load_bundle(args.out_dir / "index.pkl.gz")

    chunks: list[np.ndarray] = []
    async with _voyage_session(settings) as voyage:
        pbar = tqdm(total=len(bundle.cases), desc="voyage embed", unit="doc")
        try:
            for start in range(0, len(bundle.cases), args.batch_size):
                batch = bundle.cases[start : start + args.batch_size]
                texts = [_truncate(c.search_text()) for c in batch]
                matrix = await voyage.embed(texts, input_type="document")
                chunks.append(matrix)
                pbar.update(len(batch))
        finally:
            pbar.close()
    save_embeddings(np.vstack(chunks).astype(np.float32), args.out_dir / "embeddings.npy")

    bundle.voyage_model = settings.voyage_model
    from app.storage.index_loader import save_bundle  # localized — лень опять импортировать сверху

    save_bundle(bundle, args.out_dir / "index.pkl.gz")
    logger.info("done out_dir=%s docs=%d", args.out_dir, len(bundle.cases))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
