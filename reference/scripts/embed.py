#!/usr/bin/env python3
"""
Построение семантических эмбеддингов корпуса через Voyage AI.

Дополняет существующий BM25-индекс (data/index.pkl) семантическим слоем:
для каждого документа считается 1024-мерный вектор voyage-3-large по
объединённому тексту "заголовок + позиция ВС + фабула".

Запуск:
  export VOYAGE_API_KEY=<your-key>
  python3 scripts/embed.py
  python3 scripts/embed.py --rebuild       # пересчитать всё
  python3 scripts/embed.py --model voyage-3 # более дешёвая модель

Выход:
  data/embeddings.npy  — массив (N, dim) float32, нормализованный
  data/embeddings_meta.json  — модель, размерность, статистика
"""

import argparse
import json
import os
import pickle
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
DEFAULT_MODEL = "voyage-3-large"
BATCH_SIZE = 32
MAX_RETRIES = 4
PARALLEL_WORKERS = 3


def doc_to_embedding_text(doc: dict) -> str:
    """Готовит текст для эмбеддинга: заголовок + позиция ВС + фабула.

    Если структура не парсится, используется полный текст с разумным усечением.
    Цель — концентрированно передать смысл обзора, не утопая в шумном контексте.
    """
    parts = [doc["title"]]
    sec = doc["sections"]
    if sec["vs_position"]:
        parts.append("Позиция ВС: " + sec["vs_position"])
    if sec["fabula"]:
        parts.append("Фабула: " + sec["fabula"])
    if not sec["vs_position"] and not sec["fabula"]:
        body = sec["residual"] or doc["text"]
        parts.append(body[:8000])
    return "\n".join(parts)


def embed_batch(texts: list, model: str, api_key: str, input_type: str = "document"):
    """Возвращает (эмбеддинги, usage). Повторяет при временных ошибках."""
    payload = json.dumps({
        "input": texts,
        "model": model,
        "input_type": input_type,
    }).encode()

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(
                VOYAGE_URL,
                data=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.loads(r.read())
            return [item["embedding"] for item in resp["data"]], resp.get("usage", {})
        except urllib.error.HTTPError as e:
            err_body = e.read().decode(errors="replace")
            last_err = f"HTTP {e.code}: {err_body[:300]}"
            if e.code == 429 or e.code >= 500:
                wait = 2 ** attempt
                print(f"    {last_err}. Жду {wait}s, повтор...", file=sys.stderr)
                time.sleep(wait)
                continue
            raise RuntimeError(last_err)
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = f"Network: {e}"
            time.sleep(2 ** attempt)
            continue
    raise RuntimeError(f"После {MAX_RETRIES} попыток: {last_err}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--meta", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Только первые N (отладка)")
    args = parser.parse_args()

    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        print("ERROR: переменная окружения VOYAGE_API_KEY не установлена.", file=sys.stderr)
        print("Выполните: export VOYAGE_API_KEY=<your-key>", file=sys.stderr)
        sys.exit(1)

    here = Path(__file__).parent.parent
    if args.index:
        index_path = Path(args.index)
    else:
        gz = here / "data" / "index.pkl.gz"
        plain = here / "data" / "index.pkl"
        index_path = gz if gz.exists() else plain
    out_path = Path(args.out) if args.out else here / "data" / "embeddings.npy"
    meta_path = Path(args.meta) if args.meta else here / "data" / "embeddings_meta.json"

    if not index_path.exists():
        print(f"ERROR: BM25-индекс не найден: {index_path}", file=sys.stderr)
        print("Сначала выполните: python3 scripts/index.py", file=sys.stderr)
        sys.exit(1)

    if out_path.exists() and not args.rebuild:
        print(f"Эмбеддинги уже существуют: {out_path}")
        print("Используйте --rebuild для пересчёта")
        sys.exit(0)

    print(f"Загружаю BM25-индекс из {index_path}...")
    if str(index_path).endswith(".gz"):
        import gzip
        with gzip.open(index_path, "rb") as f:
            idx = pickle.load(f)
    else:
        with open(index_path, "rb") as f:
            idx = pickle.load(f)
    docs = idx["documents"]
    if args.limit:
        docs = docs[: args.limit]
    print(f"  {len(docs)} документов")

    print(f"Готовлю тексты для эмбеддинга...")
    texts = [doc_to_embedding_text(d) for d in docs]
    avg_len = sum(len(t) for t in texts) // len(texts)
    print(f"  Средняя длина: {avg_len} символов")

    batches = []
    for i in range(0, len(texts), BATCH_SIZE):
        batches.append((i, texts[i : i + BATCH_SIZE]))
    print(f"  Батчей: {len(batches)} по {BATCH_SIZE}")

    print(f"\nЭмбеддинг через {args.model} ({PARALLEL_WORKERS} потоков)...")
    embeddings = [None] * len(texts)
    total_tokens = 0
    completed = 0
    started = time.time()

    def process(batch_info):
        idx_start, batch_texts = batch_info
        embs, usage = embed_batch(batch_texts, args.model, api_key)
        return idx_start, embs, usage.get("total_tokens", 0)

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
        futures = [ex.submit(process, b) for b in batches]
        for f in as_completed(futures):
            idx_start, embs, tokens = f.result()
            for j, e in enumerate(embs):
                embeddings[idx_start + j] = e
            total_tokens += tokens
            completed += 1
            if completed % 10 == 0 or completed == len(batches):
                elapsed = time.time() - started
                rate = completed / elapsed
                eta = (len(batches) - completed) / rate if rate > 0 else 0
                print(f"  {completed}/{len(batches)} батчей, "
                      f"токенов: {total_tokens:,}, "
                      f"elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s")

    arr = np.array(embeddings, dtype=np.float32)
    print(f"\nМассив: {arr.shape}, dtype={arr.dtype}")

    # Нормализация: cosine similarity = dot product нормализованных векторов
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr_normalized = arr / np.where(norms > 0, norms, 1.0)

    np.save(out_path, arr_normalized)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Сохранено: {out_path} ({size_mb:.1f} МБ)")

    elapsed = time.time() - started
    # Цены Voyage AI на момент сборки
    price_per_million = {
        "voyage-3-large": 0.18,
        "voyage-3": 0.06,
        "voyage-3-lite": 0.02,
    }.get(args.model, 0.18)
    cost = total_tokens * price_per_million / 1_000_000

    meta = {
        "model": args.model,
        "dimension": int(arr.shape[1]),
        "documents": int(arr.shape[0]),
        "normalized": True,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(cost, 4),
        "elapsed_seconds": round(elapsed, 1),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nИтого:")
    print(f"  Время: {elapsed:.1f}s")
    print(f"  Токенов: {total_tokens:,}")
    print(f"  Стоимость: ~${cost:.3f}")


if __name__ == "__main__":
    main()
