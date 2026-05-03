"""Однократная индексация корпуса.

Объединяет логику reference/scripts/index.py и reference/scripts/embed.py:
1. Парсит court_practice_db.json (исходный JSON-дамп Telegram-каналов).
2. Разбивает каждый обзор на секции (`fabula`, `lower_courts`, `vs_position`, `residual`).
3. Лемматизирует через pymorphy3 + кэш.
4. Нормализует case_id для дедупликации (regex-парсер трёх форматов).
5. Парсит дату в ISO.
6. Строит 5 BM25-индексов по секциям.
7. (опционально) Считает эмбеддинги через Voyage AI и сохраняет как float16
   нормализованные — совместимо с reference/data/embeddings.npy.
8. Сохраняет в формате, совместимом с reference: index.pkl.gz и embeddings.npy.

Запуск (локально):
    python -m scripts.build_index --source data/court_practice_db.json --out-dir data
    python -m scripts.build_index --source data/court_practice_db.json --out-dir data --embeddings

На Railway:
    railway run python -m scripts.build_index \
        --source /data/court_practice_db.json --out-dir /data --embeddings
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from app.config import get_settings
from app.search.bm25 import SECTION_FIELDS
from app.search.lemmatizer import Lemmatizer
from app.search.semantic import VoyageClient
from app.storage.index_loader import save_bundle_dict, save_embeddings


logger = logging.getLogger("build_index")


# ============================================================================
# Парсинг секций обзора (1-в-1 c reference/scripts/index.py)
# ============================================================================

SECTION_MARKERS: dict[str, list[str]] = {
    "fabula": [
        r"Фабула дела[:\s]*",
        r"Обстоятельства дела[:\s]*",
    ],
    "lower_courts": [
        r"Постановления судов[:\s]*",
        r"Судебные акты[:\s]*",
        r"Позиция нижестоящих судов?[:\s]*",
        r"Постановления нижестоящих судов?[:\s]*",
    ],
    "vs_position": [
        r"Позиция Верховного [Сс]уда[:\s]*",
        r"Позиция ВС[:\s]*",
        r"Верховный [Сс]уд указал[аои]?[:\s]*",
        r"СК[ГЭ]С указал[аои]?[:\s]*",
        r"Выводы Верховного [Сс]уда[:\s]*",
    ],
}


def parse_sections(text: str) -> dict[str, str]:
    sections = {"fabula": "", "lower_courts": "", "vs_position": "", "residual": ""}
    found: list[tuple[int, int, str]] = []
    for sec_name, patterns in SECTION_MARKERS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                found.append((match.start(), match.end(), sec_name))
    if not found:
        sections["residual"] = text
        return sections
    found.sort()
    for i, (_, end_pos, sec_name) in enumerate(found):
        next_start = found[i + 1][0] if i + 1 < len(found) else len(text)
        chunk = text[end_pos:next_start].strip()
        if sections[sec_name]:
            sections[sec_name] += "\n" + chunk
        else:
            sections[sec_name] = chunk
    if found[0][0] > 0:
        sections["residual"] = text[: found[0][0]].strip()
    return sections


def normalize_case_id(case_number: str) -> str:
    """Канонический id для дедупа. Три формата: ВС-кассация, арбитраж, гражданское."""
    if not case_number:
        return ""
    text = re.sub(r"\s+", " ", case_number).strip()
    vs_match = re.search(r"\b(\d{2,3})[\s-]?[ЭЭКк][СсГг][\s-]?(\d{1,2})[\s-]?(\d{2,6})", text)
    if vs_match:
        return f"{vs_match.group(1)}-{vs_match.group(2)}-{vs_match.group(3)}"
    arb_match = re.search(r"\bА\s*(\d{1,3})[\s-](\d{1,7})\s*/\s*(\d{4})", text)
    if arb_match:
        return f"А{arb_match.group(1)}-{arb_match.group(2)}/{arb_match.group(3)}"
    civ_match = re.search(r"\b(\d{1,3})[\s-]?КГ[\s-]?(\d{1,2})[\s-]?(\d{1,6})", text)
    if civ_match:
        return f"{civ_match.group(1)}-КГ{civ_match.group(2)}-{civ_match.group(3)}"
    return ""


def parse_date(date_str: str) -> str:
    if not date_str:
        return ""
    m = re.match(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", date_str.strip())
    if not m:
        return ""
    try:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return datetime(y, mo, d).date().isoformat()
    except (ValueError, OverflowError):
        return ""


# ============================================================================
# Текст для эмбеддинга
# ============================================================================


def doc_to_embedding_text(doc: dict[str, Any]) -> str:
    """title + Позиция ВС + Фабула. Если структуры нет — residual или text[:8000]."""
    parts = [doc.get("title", "")]
    sec = doc["sections"]
    if sec["vs_position"]:
        parts.append("Позиция ВС: " + sec["vs_position"])
    if sec["fabula"]:
        parts.append("Фабула: " + sec["fabula"])
    if not sec["vs_position"] and not sec["fabula"]:
        body = sec["residual"] or doc.get("text", "")
        parts.append(body[:8000])
    return "\n".join(p for p in parts if p)


# ============================================================================
# BM25-индексация
# ============================================================================


def enrich_documents(raw: list[dict[str, Any]], lem: Lemmatizer) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    section_stats = {"fabula": 0, "lower_courts": 0, "vs_position": 0, "residual_only": 0}

    for msg in tqdm(raw, desc="enrich", unit="doc"):
        sections = parse_sections(msg.get("text", ""))
        if sections["vs_position"]:
            section_stats["vs_position"] += 1
        if sections["fabula"]:
            section_stats["fabula"] += 1
        if sections["lower_courts"]:
            section_stats["lower_courts"] += 1
        if sections["residual"] and not (sections["fabula"] or sections["vs_position"]):
            section_stats["residual_only"] += 1

        title = msg.get("title", "")
        title_lemmas = lem.tokenize(title)
        fabula_lemmas = lem.tokenize(sections["fabula"])
        lower_lemmas = lem.tokenize(sections["lower_courts"])
        vs_lemmas = lem.tokenize(sections["vs_position"])
        residual_lemmas = lem.tokenize(sections["residual"])

        # Если структуры нет — residual = vs_position (как в reference, чтобы не терять сигнал).
        if not vs_lemmas and not fabula_lemmas:
            vs_lemmas = residual_lemmas

        full_lemmas = title_lemmas + fabula_lemmas + lower_lemmas + vs_lemmas + residual_lemmas

        tag_lemmas: list[str] = []
        for tag in msg.get("hashtags", []):
            tag_lemmas.extend(lem.tokenize(tag))

        case_id = normalize_case_id(msg.get("case_number", ""))
        iso_date = parse_date(msg.get("date", ""))
        year = int(iso_date[:4]) if iso_date else None

        enriched.append({
            "id": msg["id"],
            "court": msg.get("court", ""),
            "date": msg.get("date", ""),
            "iso_date": iso_date,
            "year": year,
            "title": title,
            "case_number": msg.get("case_number", ""),
            "case_id": case_id,
            "text": msg.get("text", ""),
            "hashtags": msg.get("hashtags", []),
            "articles": msg.get("articles", []),
            "source_channel": msg.get("source_channel", ""),
            "sections": sections,
            "lemmas": {
                "title": title_lemmas,
                "fabula": fabula_lemmas,
                "lower_courts": lower_lemmas,
                "vs_position": vs_lemmas,
                "full": full_lemmas,
                "tags": tag_lemmas,
            },
        })

    logger.info("section_stats %s", section_stats)
    return enriched


def build_bm25(enriched: list[dict[str, Any]]) -> dict[str, BM25Okapi]:
    out: dict[str, BM25Okapi] = {}
    for field in SECTION_FIELDS:
        corpus = [doc["lemmas"][field] or [""] for doc in enriched]
        out[field] = BM25Okapi(corpus)
        logger.info("bm25_built field=%s", field)
    return out


def build_case_groups(enriched: list[dict[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for doc in enriched:
        cid = doc["case_id"]
        if cid:
            groups.setdefault(cid, []).append(doc["id"])
    return groups


# ============================================================================
# Эмбеддинги
# ============================================================================


@asynccontextmanager
async def _voyage_session(api_key: str, model: str, timeout_s: float):  # type: ignore[no-untyped-def]
    client = VoyageClient(api_key=api_key, model=model, timeout_s=timeout_s)
    try:
        yield client
    finally:
        await client.aclose()


async def build_embeddings(
    enriched: list[dict[str, Any]],
    voyage: VoyageClient,
    batch_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    pbar = tqdm(total=len(enriched), desc="voyage embed", unit="doc")
    try:
        for start in range(0, len(enriched), batch_size):
            batch = enriched[start : start + batch_size]
            texts = [doc_to_embedding_text(d) for d in batch]
            matrix = await voyage.embed(texts, input_type="document")
            chunks.append(matrix)
            pbar.update(len(batch))
    finally:
        pbar.close()
    return np.vstack(chunks).astype(np.float32)


# ============================================================================
# Точка входа
# ============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, type=Path, help="Путь к court_practice_db.json")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument(
        "--embeddings",
        action="store_true",
        help="Построить эмбеддинги через Voyage (требует VOYAGE_API_KEY)",
    )
    p.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Только BM25, без Voyage (синоним отсутствия --embeddings)",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--force",
        action="store_true",
        help="Перезаписать существующие index.pkl.gz / embeddings.npy без вопросов",
    )
    return p.parse_args()


def _confirm_overwrite(path: Path) -> bool:
    if not sys.stdin.isatty():
        # Безопасно: в неинтерактивном режиме не перезатираем.
        print(f"[skip] {path} существует. Запусти с --force для перезаписи.", file=sys.stderr)
        return False
    answer = input(f"{path} уже существует. Перезаписать? [y/N] ").strip().lower()
    return answer in ("y", "yes", "д", "да")


async def amain() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = out_dir / "index.pkl.gz"
    embeddings_path = out_dir / "embeddings.npy"

    do_embeddings = args.embeddings and not args.skip_embeddings

    if index_path.exists() and not args.force:
        if not _confirm_overwrite(index_path):
            return 0
    if do_embeddings and embeddings_path.exists() and not args.force:
        if not _confirm_overwrite(embeddings_path):
            return 0

    if not args.source.exists():
        sys.exit(f"[fatal] исходный JSON не найден: {args.source}")

    logger.info("loading source=%s", args.source)
    with args.source.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, list):
        sys.exit("[fatal] ожидаю JSON-массив объектов")
    logger.info("loaded docs=%d", len(raw))

    lem = Lemmatizer()
    enriched = enrich_documents(raw, lem)
    bm25 = build_bm25(enriched)
    case_groups = build_case_groups(enriched)

    payload: dict[str, Any] = {
        "version": 1,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "documents": enriched,
        "bm25": bm25,
        "case_groups": case_groups,
        "lemma_cache": lem.cache_snapshot(),
    }
    save_bundle_dict(payload, index_path)
    logger.info(
        "index_done docs=%d unique_cases=%d cache=%d size_mb=%.1f",
        len(enriched),
        len(case_groups),
        lem.cache_size,
        index_path.stat().st_size / 1024 / 1024,
    )

    if do_embeddings:
        settings = get_settings()
        async with _voyage_session(
            settings.voyage_api_key, settings.voyage_model, settings.voyage_timeout_s
        ) as voyage:
            matrix = await build_embeddings(enriched, voyage, args.batch_size)
        # Совместимо с reference: float16 normalized.
        save_embeddings(matrix.astype(np.float16, copy=False), embeddings_path)
        logger.info(
            "embeddings_done shape=%s size_mb=%.1f",
            matrix.shape,
            embeddings_path.stat().st_size / 1024 / 1024,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
