"""Однократная индексация корпуса.

Запуск:
    python -m scripts.build_index --source data/court_practice_db.json --out-dir data
    python -m scripts.build_index --source data/court_practice_db.json --out-dir data --embeddings

На Railway:
    railway run python -m scripts.build_index --source /data/court_practice_db.json --out-dir /data --embeddings

Ожидаемая схема court_practice_db.json — список объектов:
    [
      {
        "id": 1,                              # обязательно: уникальный int
        "title": "...",                       # обязательно
        "court": "СКГД" | "СКЭС",            # обязательно
        "date": "2024-01-15",                 # ISO
        "case_number": "А40-...",
        "fabula": "...",
        "lower_courts_position": "...",
        "vs_position": "...",
        "tags": ["..."],
        "articles": ["ст. 333 ГК РФ"],
        "source_channel": "...",
        "url": "https://..."
      },
      ...
    ]

Если в твоём JSON ключи отличаются — поправь _normalize() ниже. Скрипт не падает на
отсутствующих полях, просто проставляет пустые строки.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from app.config import get_settings
from app.search.bm25 import BM25Index
from app.search.engine import Case, IndexBundle
from app.search.semantic import VoyageClient
from app.storage.index_loader import save_bundle, save_embeddings


logger = logging.getLogger("build_index")


def _normalize(raw: dict) -> Case:
    """Привести запись из JSON к Case. Здесь правь маппинг под свой формат."""
    return Case(
        id=int(raw["id"]),
        title=str(raw.get("title", "")).strip(),
        court=str(raw.get("court", "")).strip(),
        date=str(raw.get("date", "")).strip(),
        case_number=str(raw.get("case_number", "")).strip(),
        fabula=str(raw.get("fabula", "")).strip(),
        lower_courts_position=str(raw.get("lower_courts_position", "")).strip(),
        vs_position=str(raw.get("vs_position", "")).strip(),
        tags=[str(t).strip() for t in raw.get("tags", []) if str(t).strip()],
        articles=[str(a).strip() for a in raw.get("articles", []) if str(a).strip()],
        source_channel=str(raw.get("source_channel", "")).strip(),
        url=raw.get("url"),
    )


def _read_corpus(path: Path) -> list[Case]:
    if not path.exists():
        sys.exit(f"[fatal] исходный JSON не найден: {path}")
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, list):
        sys.exit("[fatal] ожидаю JSON-массив объектов")
    cases: list[Case] = []
    seen: set[int] = set()
    for i, item in enumerate(raw):
        try:
            case = _normalize(item)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("skip_bad_record", extra={"index": i, "err": str(exc)})
            continue
        if case.id in seen:
            logger.warning("duplicate_id_skipped", extra={"id": case.id})
            continue
        seen.add(case.id)
        cases.append(case)
    cases.sort(key=lambda c: c.id)
    logger.info("corpus_loaded size=%d", len(cases))
    return cases


def _corpus_hash(cases: list[Case]) -> str:
    h = hashlib.sha256()
    for c in cases:
        h.update(c.search_text().encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


async def _build_embeddings(
    cases: list[Case],
    voyage: VoyageClient,
    batch_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    pbar = tqdm(total=len(cases), desc="voyage embed", unit="doc")
    try:
        for start in range(0, len(cases), batch_size):
            batch = cases[start : start + batch_size]
            texts = [_truncate(c.search_text()) for c in batch]
            mat = await voyage.embed(texts, input_type="document")
            chunks.append(mat)
            pbar.update(len(batch))
    finally:
        pbar.close()
    return np.vstack(chunks).astype(np.float32)


def _truncate(text: str, max_chars: int = 30_000) -> str:
    """Voyage держит ~32к токенов на документ, поэтому ограничиваем по символам."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build BM25 index (and optional embeddings).")
    p.add_argument("--source", required=True, type=Path, help="Путь к court_practice_db.json")
    p.add_argument("--out-dir", required=True, type=Path, help="Куда сохранять index/embeddings")
    p.add_argument(
        "--embeddings",
        action="store_true",
        help="Также построить эмбеддинги через Voyage (требует VOYAGE_API_KEY)",
    )
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


async def amain() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = _read_corpus(args.source)
    if not cases:
        sys.exit("[fatal] корпус пуст")

    docs = [c.search_text() for c in cases]
    logger.info("building_bm25 docs=%d", len(docs))
    bm25 = BM25Index.build(docs)

    tag_counts: Counter[str] = Counter(t for c in cases for t in c.tags)
    article_counts: Counter[str] = Counter(a for c in cases for a in c.articles)

    voyage_model = ""
    if args.embeddings:
        settings = get_settings()
        voyage_model = settings.voyage_model
        async with _voyage_session(settings) as voyage:
            matrix = await _build_embeddings(cases, voyage, args.batch_size)
        save_embeddings(matrix, out_dir / "embeddings.npy")

    bundle = IndexBundle(
        cases=cases,
        bm25=bm25,
        tag_counts=tag_counts,
        article_counts=article_counts,
        built_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        corpus_hash=_corpus_hash(cases),
        voyage_model=voyage_model,
    )
    save_bundle(bundle, out_dir / "index.pkl.gz")
    logger.info("done out_dir=%s embeddings=%s", out_dir, args.embeddings)
    return 0


from contextlib import asynccontextmanager  # noqa: E402  — bottom-of-file для читаемости


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


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
