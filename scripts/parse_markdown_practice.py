"""Парсер markdown-дайджеста свежей практики ВС РФ → JSON-документы в схеме court_practice_db.json.

Markdown собирается из четырёх Telegram-каналов и имеет жёсткую структуру:

    ## <Коллегия> (N)


    ### <Номер>. DD.MM.YYYY — <Title>


    **Реквизиты:** <case_number>     # опционально
    **Теги:** #tag1, #tag2            # опционально
    **Нормы:** ст. 31 ЖК; ст. 12 ГК   # опционально
    **Источник:** канал «<Name>»
    **Текст обзора:**

    <full text>

    ---

На выходе — список dict-ов, совместимых с тем, что `scripts/build_index.py` принимает
из `court_practice_db.json` (см. `enrich_documents`).

Запуск:
    python -m scripts.parse_markdown_practice \\
        --source "Свежая_практика_ВС_РФ_после_21_03_2026.md" \\
        --out data/new_practice_after_2026_03_21.json \\
        --start-id 5347
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


# Заголовки коллегий (level-2). Маппим в значение поля `court`.
COURT_HEADER_MAP: dict[str, str] = {
    "СКГД ВС РФ": "СКГД ВС РФ",
    "СКЭС ВС РФ": "СКЭС ВС РФ",
    "ВС РФ": "ВС РФ",
}


# Каналы — приводим к строке без префикса "канал «...»", как уже хранится в существующих документах.
SOURCE_CHANNEL_MAP: dict[str, str] = {
    "СКГД/СКЭС (original)": "СКГД/СКЭС (original)",
    "Новости ВС РФ": "Новости ВС РФ",
    "ЭКОНОМКОЛЛЕГИЯ FRESH": "ЭКОНОМКОЛЛЕГИЯ FRESH",
}


_COURT_HEADER_RE = re.compile(r"^##\s+([^\n(]+?)\s*\(\s*\d+\s*\)\s*$")
_CASE_HEADER_RE = re.compile(
    r"^###\s+\d+\.\s+(\d{1,2}\.\d{1,2}\.\d{4})\s+[—-]\s+(.+?)\s*$"
)
_FIELD_RE = re.compile(r"^\*\*(Реквизиты|Теги|Нормы|Источник|Текст обзора):\*\*\s*(.*)$")
_CHANNEL_INLINE_RE = re.compile(r"канал\s+«([^»]+)»")


@dataclass
class CaseBlock:
    court: str
    date: str            # DD.MM.YYYY (из заголовка)
    title: str
    case_number: str = ""
    hashtags: list[str] = field(default_factory=list)
    articles: list[str] = field(default_factory=list)
    source_channel: str = ""
    text: str = ""


def _split_csv(value: str) -> list[str]:
    """`#a, #b` → ['a', 'b']; `ст. 1 ГК; ст. 2 ГК` → ['ст. 1 ГК', 'ст. 2 ГК']."""
    if not value:
        return []
    parts = re.split(r"[;,]", value)
    return [p.strip() for p in parts if p.strip()]


def _clean_hashtags(items: list[str]) -> list[str]:
    """Снимаем `#` и нижний регистр: `#ВСрешил` → `ВСрешил`."""
    out: list[str] = []
    for item in items:
        item = item.lstrip("#").strip()
        if item:
            out.append(item)
    return out


def _parse_channel(raw_source: str) -> str:
    """`канал «Новости ВС РФ»` → `Новости ВС РФ`. Если не парсится — возвращаем как есть."""
    m = _CHANNEL_INLINE_RE.search(raw_source)
    if not m:
        return raw_source.strip()
    name = m.group(1).strip()
    return SOURCE_CHANNEL_MAP.get(name, name)


def parse_markdown(md_path: Path) -> list[CaseBlock]:
    """Стейт-машина по строкам: фиксируем текущую коллегию + текущий case + текст обзора.

    Внутри секции "Текст обзора:" складываем всё до следующего `### ` или `---`.
    """
    current_court: str | None = None
    cases: list[CaseBlock] = []
    cur: CaseBlock | None = None

    # State for "Текст обзора:" body collection
    in_body = False
    body_lines: list[str] = []

    def flush_body() -> None:
        if cur is None:
            return
        # Сжимаем последовательности пустых строк до одной — как принято в исходном корпусе.
        text = "\n".join(body_lines).strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        cur.text = text

    def close_case() -> None:
        nonlocal cur, in_body, body_lines
        if cur is None:
            return
        flush_body()
        cases.append(cur)
        cur = None
        in_body = False
        body_lines = []

    with md_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")

            # Заголовок коллегии (##) — старт новой секции.
            m_court = _COURT_HEADER_RE.match(line)
            if m_court:
                close_case()
                name = m_court.group(1).strip()
                current_court = COURT_HEADER_MAP.get(name)
                continue

            # Заголовок кейса (###)
            m_case = _CASE_HEADER_RE.match(line)
            if m_case and current_court:
                close_case()
                cur = CaseBlock(
                    court=current_court,
                    date=m_case.group(1),
                    title=m_case.group(2).strip(),
                )
                continue

            if cur is None:
                continue

            # Разделитель `---` всегда закрывает текущий кейс.
            if line.strip() == "---":
                close_case()
                continue

            # Поля **Field:** value
            m_field = _FIELD_RE.match(line)
            if m_field:
                key, value = m_field.group(1), m_field.group(2).strip()
                if key == "Реквизиты":
                    cur.case_number = value
                elif key == "Теги":
                    cur.hashtags = _clean_hashtags(_split_csv(value))
                elif key == "Нормы":
                    cur.articles = _split_csv(value)
                elif key == "Источник":
                    cur.source_channel = _parse_channel(value)
                elif key == "Текст обзора":
                    in_body = True
                    body_lines = []
                continue

            if in_body:
                body_lines.append(line)

    close_case()
    return cases


def case_to_db_record(case: CaseBlock, doc_id: int) -> dict[str, object]:
    """Привести CaseBlock к dict в схеме court_practice_db.json (без секций/лемм —
    их посчитает build_index при индексации)."""
    return {
        "id": doc_id,
        "court": case.court,
        "date": case.date,
        "title": case.title,
        "case_number": case.case_number,
        "text": case.text,
        "hashtags": case.hashtags,
        "articles": case.articles,
        "source_channel": case.source_channel,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, type=Path, help="Путь к markdown-дайджесту")
    p.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Куда писать JSON-список новых документов (схема court_practice_db.json)",
    )
    p.add_argument(
        "--start-id",
        type=int,
        required=True,
        help="С какого id начинать нумерацию новых записей (обычно max(existing_ids) + 1)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cases = parse_markdown(args.source)
    records = [case_to_db_record(c, args.start_id + i) for i, c in enumerate(cases)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
    print(
        f"parsed={len(records)} ids={args.start_id}..{args.start_id + len(records) - 1} → {args.out}"
    )


if __name__ == "__main__":
    main()
