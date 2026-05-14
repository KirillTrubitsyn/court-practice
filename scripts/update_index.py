"""Инкрементальное обновление индекса свежими документами.

Зачем:
    `build_index.py` строит индекс с нуля из полного `court_practice_db.json`. Этот
    скрипт нужен, когда полного дампа корпуса под рукой нет (он живёт на Railway-
    Volume), а добавить надо только пачку новых записей.

Что делает:
    1. Загружает существующий `index.pkl.gz`.
    2. Загружает JSON-список новых документов в схеме `court_practice_db.json`.
    3. Прогоняет каждый новый документ через ту же логику, что и `build_index`:
       parse_sections → лемматизация → нормализация case_id → дата → ISO.
    4. Аппендит новые documents к существующим, заново строит BM25 по всему
       корпусу (rank_bm25 не поддерживает инкремент), пересобирает case_groups,
       расширяет lemma_cache, сохраняет `index.pkl.gz` атомарно.
    5. Если `--embeddings` — считает эмбеддинги Voyage для новых документов
       (`input_type=document`, тот же текст, что в `doc_to_embedding_text`),
       приводит к float16 (как в reference), VSTACK к существующему
       `embeddings.npy`, сохраняет.

Запуск:
    # Только BM25 (без эмбеддингов — для офлайн-проверки / dry-run):
    python -m scripts.update_index \\
        --new data/new_practice_after_2026_03_21.json \\
        --index data/index.pkl.gz

    # С эмбеддингами (требует VOYAGE_API_KEY в env / .env):
    python -m scripts.update_index \\
        --new data/new_practice_after_2026_03_21.json \\
        --index data/index.pkl.gz \\
        --embeddings data/embeddings.npy
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from app.search.bm25 import SECTION_FIELDS
from app.search.lemmatizer import Lemmatizer
from scripts.build_index import (
    build_case_groups,
    doc_to_embedding_text,
    extract_case_ids,
    parse_date,
    parse_sections,
)


logger = logging.getLogger("update_index")


def _load_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"[fatal] индекс не найден: {path}")
    with gzip.open(path, "rb") as fh:
        data: dict[str, Any] = pickle.load(fh)
    if not isinstance(data, dict) or "documents" not in data or "bm25" not in data:
        raise SystemExit(f"[fatal] неожиданная структура индекса в {path}")
    return data


def _save_index_atomic(payload: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wb", compresslevel=6) as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _enrich_one(raw: dict[str, Any], lem: Lemmatizer) -> dict[str, Any]:
    sections = parse_sections(raw.get("text", ""))

    title = raw.get("title", "")
    title_lemmas = lem.tokenize(title)
    fabula_lemmas = lem.tokenize(sections["fabula"])
    lower_lemmas = lem.tokenize(sections["lower_courts"])
    vs_lemmas = lem.tokenize(sections["vs_position"])
    residual_lemmas = lem.tokenize(sections["residual"])

    # Если структуры нет — residual считаем vs_position (как в reference / build_index).
    if not vs_lemmas and not fabula_lemmas:
        vs_lemmas = residual_lemmas

    full_lemmas = title_lemmas + fabula_lemmas + lower_lemmas + vs_lemmas + residual_lemmas

    tag_lemmas: list[str] = []
    for tag in raw.get("hashtags", []):
        tag_lemmas.extend(lem.tokenize(tag))

    case_ids = extract_case_ids(raw.get("case_number", ""), raw.get("text", ""))
    case_id = case_ids[0] if case_ids else ""
    iso_date = parse_date(raw.get("date", ""))
    year = int(iso_date[:4]) if iso_date else None

    return {
        "id": raw["id"],
        "court": raw.get("court", ""),
        "date": raw.get("date", ""),
        "iso_date": iso_date,
        "year": year,
        "title": title,
        "case_number": raw.get("case_number", ""),
        "case_id": case_id,
        "case_ids": case_ids,
        "text": raw.get("text", ""),
        "hashtags": raw.get("hashtags", []),
        "articles": raw.get("articles", []),
        "source_channel": raw.get("source_channel", ""),
        "sections": sections,
        "lemmas": {
            "title": title_lemmas,
            "fabula": fabula_lemmas,
            "lower_courts": lower_lemmas,
            "vs_position": vs_lemmas,
            "full": full_lemmas,
            "tags": tag_lemmas,
        },
    }


def _rebuild_bm25(documents: list[dict[str, Any]]) -> dict[str, BM25Okapi]:
    out: dict[str, BM25Okapi] = {}
    for field in SECTION_FIELDS:
        corpus = [doc["lemmas"][field] or [""] for doc in documents]
        out[field] = BM25Okapi(corpus)
        logger.info("bm25_rebuilt field=%s", field)
    return out


async def _build_embeddings_for_new(
    new_docs: list[dict[str, Any]],
    api_key: str,
    model: str,
    batch_size: int,
) -> np.ndarray:
    from app.search.semantic import VoyageClient

    client = VoyageClient(api_key=api_key, model=model)
    try:
        chunks: list[np.ndarray] = []
        pbar = tqdm(total=len(new_docs), desc="voyage embed", unit="doc")
        for start in range(0, len(new_docs), batch_size):
            batch = new_docs[start : start + batch_size]
            texts = [doc_to_embedding_text(d) for d in batch]
            matrix = await client.embed(texts, input_type="document")
            chunks.append(matrix)
            pbar.update(len(batch))
        pbar.close()
    finally:
        await client.aclose()
    return np.vstack(chunks).astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--new", required=True, type=Path, help="JSON со списком новых документов (схема court_practice_db.json)")
    p.add_argument("--index", required=True, type=Path, help="Путь к существующему index.pkl.gz (будет переписан)")
    p.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Путь к embeddings.npy. Если указан — достроим эмбеддинги для новых документов через Voyage.",
    )
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


async def amain() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    logger.info("loading existing index from %s", args.index)
    bundle = _load_index(args.index)
    existing_docs: list[dict[str, Any]] = bundle["documents"]
    existing_ids = {d["id"] for d in existing_docs}
    logger.info("existing docs=%d unique_case_groups=%d", len(existing_docs), len(bundle.get("case_groups") or {}))

    logger.info("loading new documents from %s", args.new)
    with args.new.open(encoding="utf-8") as fh:
        new_raw = json.load(fh)
    if not isinstance(new_raw, list):
        raise SystemExit("[fatal] --new должен быть JSON-массивом")

    # Гарантируем уникальность id новых документов.
    collisions = [r["id"] for r in new_raw if r["id"] in existing_ids]
    if collisions:
        raise SystemExit(f"[fatal] коллизия id с существующими документами: {collisions[:10]}…")

    lem = Lemmatizer(preload_cache=bundle.get("lemma_cache"))
    new_enriched = [_enrich_one(r, lem) for r in tqdm(new_raw, desc="enrich", unit="doc")]
    logger.info("enriched new=%d cache_size=%d", len(new_enriched), lem.cache_size)

    merged_docs = existing_docs + new_enriched
    bm25 = _rebuild_bm25(merged_docs)
    case_groups = build_case_groups(merged_docs)
    logger.info(
        "merged docs=%d unique_cases=%d (prev=%d)",
        len(merged_docs),
        len(case_groups),
        len(bundle.get("case_groups") or {}),
    )

    payload: dict[str, Any] = {
        "version": int(bundle.get("version", 1)),
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "documents": merged_docs,
        "bm25": bm25,
        "case_groups": case_groups,
        "lemma_cache": lem.cache_snapshot(),
    }
    _save_index_atomic(payload, args.index)
    size_mb = args.index.stat().st_size / 1024 / 1024
    logger.info("index_saved path=%s size_mb=%.1f", args.index, size_mb)

    if args.embeddings is not None:
        from app.config import get_settings

        settings = get_settings()
        existing_matrix = (
            np.load(args.embeddings, allow_pickle=False) if args.embeddings.exists() else None
        )
        if existing_matrix is not None and existing_matrix.shape[0] != len(existing_docs):
            raise SystemExit(
                f"[fatal] embeddings.npy содержит {existing_matrix.shape[0]} векторов, "
                f"а в индексе {len(existing_docs)} существующих документов"
            )

        new_matrix = await _build_embeddings_for_new(
            new_enriched, settings.voyage_api_key, settings.voyage_model, args.batch_size
        )
        new_matrix_fp16 = new_matrix.astype(np.float16, copy=False)
        if existing_matrix is not None:
            full = np.vstack([existing_matrix, new_matrix_fp16])
        else:
            full = new_matrix_fp16
        # np.save сам дописывает .npy, поэтому пишем в файл без расширения
        # и потом переименовываем — иначе временное имя получает лишнее .npy.
        tmp = args.embeddings.with_suffix(".tmp")
        np.save(tmp, full, allow_pickle=False)
        # np.save мог дописать .npy если у `tmp` его не было; нормализуем имя.
        produced = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + ".npy")
        produced.replace(args.embeddings)
        logger.info(
            "embeddings_saved path=%s shape=%s dtype=%s",
            args.embeddings,
            list(full.shape),
            full.dtype,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
