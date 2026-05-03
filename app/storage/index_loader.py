"""I/O для индекса: dump/load IndexBundle и mmap-загрузка эмбеддингов."""

from __future__ import annotations

import gzip
import logging
import pickle
from pathlib import Path

import numpy as np

from app.search.engine import IndexBundle


logger = logging.getLogger(__name__)


# pickle.HIGHEST_PROTOCOL даст 5-й — он умеет out-of-band buffers,
# но для нашего размера это излишне. Фиксируем 4 для стабильности между минорами.
PICKLE_PROTOCOL = 4


class IndexNotFoundError(FileNotFoundError):
    """Индекс ещё не построен — нужно запустить scripts/build_index.py."""


def save_bundle(bundle: IndexBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wb", compresslevel=6) as fh:
        pickle.dump(bundle, fh, protocol=PICKLE_PROTOCOL)
    tmp.replace(path)
    logger.info("index_saved", extra={"path": str(path), "bytes": path.stat().st_size})


def load_bundle(path: Path) -> IndexBundle:
    if not path.exists():
        raise IndexNotFoundError(
            f"Индекс не найден: {path}. "
            "Запусти scripts/build_index.py и убедись, что DATA_DIR указывает на Volume."
        )
    with gzip.open(path, "rb") as fh:
        bundle = pickle.load(fh)
    if not isinstance(bundle, IndexBundle):
        raise TypeError(f"Ожидаю IndexBundle, в файле {type(bundle)!r}")
    logger.info(
        "index_loaded",
        extra={
            "path": str(path),
            "size": len(bundle.cases),
            "built_at": bundle.built_at,
        },
    )
    return bundle


def save_embeddings(matrix: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.save(tmp, matrix, allow_pickle=False)
    tmp.replace(path)
    logger.info(
        "embeddings_saved",
        extra={"path": str(path), "shape": list(matrix.shape), "dtype": str(matrix.dtype)},
    )


def load_embeddings(path: Path) -> np.ndarray | None:
    """mmap-чтение, чтобы не держать всю матрицу в RAM сразу."""
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
