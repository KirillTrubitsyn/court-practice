"""Однократная миграция: исправить year артефакты, основанные на case_number.

Парсер дат подцепляет первое попадание DD.MM.YYYY в тексте, что часто даёт
старые годы (1886, 1998, 2003) для современных определений. Этот скрипт
проходит по всем документам, извлекает год из case_number и при расхождении
> 2 лет правит doc["year"] на надёжный из case_number.

iso_date и сырой `date` не трогаем — это «как Telegram прислал».

Запуск:
    python -m scripts.migrate_fix_dates data/index.pkl.gz
"""

from __future__ import annotations

import gzip
import logging
import pickle
import sys
from pathlib import Path

from app.search.date_utils import reconcile_year, year_from_anchor, year_from_case_number


logger = logging.getLogger("migrate-dates")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if len(sys.argv) < 2:
        print("usage: migrate_fix_dates.py <path/to/index.pkl.gz>", file=sys.stderr)
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

    cnt_fixed = 0
    cnt_anchor = 0
    cnt_case = 0
    cnt_no_signal = 0
    examples: list[tuple[int, int | None, int | None, int | None, int | None]] = []
    for doc in docs:
        old_year = doc.get("year")
        anchor_year = year_from_anchor(doc.get("text", ""))
        case_year = year_from_case_number(doc.get("case_number", ""))
        if anchor_year is None and case_year is None:
            cnt_no_signal += 1
            continue
        new_year = reconcile_year(old_year, anchor_year, case_year)
        if anchor_year is not None:
            cnt_anchor += 1
        elif new_year != old_year:
            cnt_case += 1
        if new_year != old_year:
            cnt_fixed += 1
            if len(examples) < 10:
                examples.append(
                    (doc.get("id", -1), old_year, anchor_year, case_year, new_year)
                )
            doc["year"] = new_year

    logger.info(
        "fixed=%d (anchor_used=%d case_used=%d no_signal=%d)",
        cnt_fixed, cnt_anchor, cnt_case, cnt_no_signal,
    )
    if examples:
        logger.info("examples (id, old_year, anchor_year, case_year, new_year):")
        for ex in examples:
            logger.info("  %s", ex)

    bundle["version"] = max(int(bundle.get("version", 1)), 3)

    tmp = path.with_suffix(path.suffix + ".tmp")
    logger.info("writing %s", tmp)
    with gzip.open(tmp, "wb", compresslevel=6) as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    logger.info("done size=%.1fMB", path.stat().st_size / 1024 / 1024)

    # Финальная сводка по year_range после фикса
    years = sorted({d["year"] for d in docs if d.get("year") and d["year"] >= 2000})
    if years:
        logger.info("year_range after fix: [%d, %d]", years[0], years[-1])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
