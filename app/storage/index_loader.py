"""I/O индекса. Совместим с pickle-форматом reference/scripts/index.py.

reference сохраняет dict со структурой:
    {
      "version": 1,
      "built_at": "ISO8601",
      "documents": [<dict per doc>],
      "bm25": {field: BM25Okapi},
      "case_groups": {case_id: [doc_id, ...]},
      "lemma_cache": {token: lemma},
    }

Мы читаем этот dict и перепаковываем `documents` в наш dataclass `Document`.
Любые недостающие поля в исходных dict превращаются в `""` или `[]`.
"""

from __future__ import annotations

import gzip
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from app.search.engine import Document, IndexBundle, Sections


logger = logging.getLogger(__name__)


class IndexNotFoundError(FileNotFoundError):
    """Индекс не построен — нужно запустить scripts/build_index.py."""


def _open_pickle(path: Path) -> Any:
    """Загрузка с автодетектом gzip — reference допускает оба формата."""
    with path.open("rb") as fh:
        magic = fh.read(2)
        fh.seek(0)
        if magic == b"\x1f\x8b":
            with gzip.open(fh, "rb") as gz:
                return pickle.load(gz)
        return pickle.load(fh)


def _to_document(raw: dict[str, Any]) -> Document:
    """Привести dict из reference-индекса к нашему Document."""
    sections_raw = raw.get("sections") or {}
    sections: Sections = {  # type: ignore[typeddict-item]
        "fabula": str(sections_raw.get("fabula", "")),
        "lower_courts": str(sections_raw.get("lower_courts", "")),
        "vs_position": str(sections_raw.get("vs_position", "")),
        "residual": str(sections_raw.get("residual", "")),
    }
    lemmas_raw = raw.get("lemmas") or {}
    return Document(
        id=int(raw["id"]),
        court=str(raw.get("court", "")),
        date=str(raw.get("date", "")),
        iso_date=str(raw.get("iso_date", "")),
        year=raw.get("year") if raw.get("year") is None else int(raw["year"]),
        title=str(raw.get("title", "")),
        case_number=str(raw.get("case_number", "")),
        case_id=str(raw.get("case_id", "")),
        text=str(raw.get("text", "")),
        hashtags=list(raw.get("hashtags", []) or []),
        articles=list(raw.get("articles", []) or []),
        source_channel=str(raw.get("source_channel", "")),
        sections=sections,
        lemmas={  # type: ignore[typeddict-item]
            "title": list(lemmas_raw.get("title", []) or []),
            "fabula": list(lemmas_raw.get("fabula", []) or []),
            "lower_courts": list(lemmas_raw.get("lower_courts", []) or []),
            "vs_position": list(lemmas_raw.get("vs_position", []) or []),
            "full": list(lemmas_raw.get("full", []) or []),
            "tags": list(lemmas_raw.get("tags", []) or []),
        },
    )


def load_bundle(path: Path) -> IndexBundle:
    if not path.exists():
        raise IndexNotFoundError(
            f"Индекс не найден: {path}. "
            "Запусти scripts/build_index.py и убедись, что DATA_DIR указывает на Volume."
        )
    raw = _open_pickle(path)
    if not isinstance(raw, dict) or "documents" not in raw or "bm25" not in raw:
        raise TypeError(
            f"Неожиданная структура индекса в {path}. "
            "Ожидаю dict с ключами documents, bm25, case_groups."
        )
    documents = [_to_document(d) for d in raw["documents"]]
    bundle = IndexBundle(
        version=int(raw.get("version", 1)),
        built_at=str(raw.get("built_at", "")),
        documents=documents,
        bm25=raw["bm25"],
        case_groups=dict(raw.get("case_groups") or {}),
        lemma_cache=dict(raw.get("lemma_cache") or {}),
    )
    logger.info(
        "index_loaded",
        extra={
            "path": str(path),
            "size": len(bundle.documents),
            "built_at": bundle.built_at,
        },
    )
    return bundle


def save_bundle_dict(payload: dict[str, Any], path: Path) -> None:
    """Сохраняем в формате reference (dict, не наш dataclass) для совместимости.

    build_index.py использует именно этот сериализатор.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wb", compresslevel=6) as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    logger.info("index_saved", extra={"path": str(path), "bytes": path.stat().st_size})


def load_embeddings(path: Path) -> np.ndarray | None:
    """mmap-чтение, чтобы не держать всю матрицу в RAM сразу.

    reference хранит float16 (нормализованные), наш SemanticIndex кастит в float32 на ходу.
    """
    if not path.exists():
        logger.warning("embeddings_missing", extra={"path": str(path)})
        return None
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    if matrix.ndim != 2:
        raise ValueError(f"embeddings.npy должен быть 2D, получил {matrix.shape}")
    logger.info(
        "embeddings_loaded",
        extra={"path": str(path), "shape": list(matrix.shape), "dtype": str(matrix.dtype)},
    )
    return matrix


def save_embeddings(matrix: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.save(tmp, matrix, allow_pickle=False)
    tmp.replace(path)
    logger.info(
        "embeddings_saved",
        extra={"path": str(path), "shape": list(matrix.shape), "dtype": str(matrix.dtype)},
    )
