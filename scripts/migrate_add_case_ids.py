"""Однократная миграция: дополнить существующий index.pkl.gz полем case_ids.

Раньше у каждого документа было одиночное поле case_id (первое попадание regex).
Это ломало дедупликацию когда одно и то же дело упоминалось в разных каналах под
разными форматами id. Новый Document.case_ids хранит ВСЕ найденные id.

Этот скрипт читает старый pickle, прогоняет extract_case_ids() по case_number+text
для каждого документа, дописывает case_ids и пересобирает case_groups. Эмбеддинги
не трогаются.

Запуск:
    python -m scripts.migrate_add_case_ids data/index.pkl.gz
"""

from __future__ import annotations

import gzip
import logging
import pickle
import sys
from collections import Counter
from pathlib import Path

import re

# Inline copy чтобы не тянуть pymorphy3/rank_bm25 при миграции.
_CASE_ID_PATTERNS = [
    (
        re.compile(r"\b(\d{2,3})[\s-]?[Ээ][Сс][\s-]?(\d{1,2})[\s-]?(\d{2,6})"),
        lambda m: f"{m.group(1)}-ЭС{m.group(2)}-{m.group(3)}",
    ),
    (
        re.compile(r"\bА\s*(\d{1,3})[\s-](\d{1,7})\s*/\s*(\d{4})"),
        lambda m: f"А{m.group(1)}-{m.group(2)}/{m.group(3)}",
    ),
    (
        re.compile(r"\b(\d{1,3})[\s-]?КГ[\s-]?(\d{1,2})[\s-]?(\d{1,6})"),
        lambda m: f"{m.group(1)}-КГ{m.group(2)}-{m.group(3)}",
    ),
]


def extract_case_ids(*texts: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for text in texts:
        if not text:
            continue
        normalized = re.sub(r"\s+", " ", text)
        for pattern, builder in _CASE_ID_PATTERNS:
            for match in pattern.finditer(normalized):
                cid = builder(match)
                if cid and cid not in seen:
                    seen.add(cid)
                    out.append(cid)
    return out


logger = logging.getLogger("migrate")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if len(sys.argv) < 2:
        print("usage: migrate_add_case_ids.py <path/to/index.pkl.gz>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"not found: {path}", file=sys.stderr)
        return 1

    logger.info("loading %s", path)
    with gzip.open(path, "rb") as fh:
        bundle = pickle.load(fh)

    docs = bundle["documents"]
    logger.info("docs=%d", len(docs))

    cnt_added = 0
    cnt_multi = 0
    multi_examples: list[tuple[int, list[str]]] = []
    for doc in docs:
        ids = extract_case_ids(doc.get("case_number", ""), doc.get("text", ""))
        doc["case_ids"] = ids
        if ids:
            cnt_added += 1
            # Перезаписываем case_id всегда: старый формат reference (без префикса ЭС)
            # несовместим с новым (305-ЭС23-29227), оставлять его смысла нет.
            doc["case_id"] = ids[0]
        if len(ids) > 1:
            cnt_multi += 1
            if len(multi_examples) < 5:
                multi_examples.append((doc.get("id", -1), ids))

    logger.info("docs_with_ids=%d (multi-id=%d)", cnt_added, cnt_multi)
    if multi_examples:
        logger.info("examples of multi-id docs:")
        for doc_id, ids in multi_examples:
            logger.info("  id=%s ids=%s", doc_id, ids)

    # Пересобрать case_groups: один документ может попасть в несколько групп.
    new_groups: dict[str, list[int]] = {}
    for doc in docs:
        for cid in doc.get("case_ids") or ([doc.get("case_id")] if doc.get("case_id") else []):
            if cid:
                new_groups.setdefault(cid, []).append(doc["id"])
    logger.info("case_groups before=%d after=%d", len(bundle.get("case_groups", {})), len(new_groups))
    bundle["case_groups"] = new_groups
    bundle["version"] = max(int(bundle.get("version", 1)), 2)

    # Атомарно перезаписать.
    tmp = path.with_suffix(path.suffix + ".tmp")
    logger.info("writing %s", tmp)
    with gzip.open(tmp, "wb", compresslevel=6) as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    logger.info("done size=%.1fMB", path.stat().st_size / 1024 / 1024)

    # Сводка дедупа
    duplicates = sum(1 for ids in new_groups.values() if len(ids) > 1)
    in_dup_groups = sum(len(ids) for ids in new_groups.values() if len(ids) > 1)
    logger.info("duplicate cases=%d records_in_dup_groups=%d", duplicates, in_dup_groups)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
