#!/usr/bin/env python3
"""
Индексатор базы судебной практики ВС РФ.

Что делает:
  1. Парсит каждый обзор на структурные секции (заголовок, фабула,
     позиции нижестоящих судов, позиция ВС РФ).
  2. Лемматизирует тексты через pymorphy3.
  3. Извлекает нормализованный номер дела для дедупликации.
  4. Парсит даты в ISO-формат.
  5. Строит BM25-индексы по секциям (title, vs_position, fabula, full).
  6. Сохраняет индекс в виде pickle-файла.

Запуск:
  python3 scripts/index.py
  python3 scripts/index.py --rebuild  # принудительное переиндексирование

Выход:
  data/index.pkl  — основной индекс
  data/index_meta.json — метаинформация (для диагностики)
"""

import subprocess
import sys


def _ensure_deps():
    """Тихая установка зависимостей, если их нет."""
    missing = []
    try:
        import pymorphy3  # noqa: F401
    except ImportError:
        missing.append("pymorphy3")
    try:
        import rank_bm25  # noqa: F401
    except ImportError:
        missing.append("rank_bm25")
    if missing:
        print(f"Устанавливаю зависимости: {', '.join(missing)}...", file=sys.stderr)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--break-system-packages", *missing],
            check=True,
        )


_ensure_deps()


import argparse
import json
import pickle
import re
from datetime import datetime
from pathlib import Path

import pymorphy3
from rank_bm25 import BM25Okapi


# Стоп-слова: общеупотребительные русские + юридический шум,
# который размывает BM25-ранжирование, если попадает в запрос
STOPWORDS = {
    # местоимения, союзы, предлоги
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а",
    "то", "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же",
    "вы", "за", "бы", "по", "только", "ее", "мне", "было", "вот", "от",
    "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда", "даже",
    "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был", "него",
    "до", "вас", "нибудь", "опять", "уж", "вам", "ведь", "там", "потом",
    "себя", "ничего", "ей", "может", "они", "тут", "где", "есть", "надо",
    "ней", "для", "мы", "тебя", "их", "чем", "была", "сам", "чтоб", "без",
    "будто", "чего", "раз", "тоже", "себе", "под", "будет", "ж", "тогда",
    "кто", "этот", "того", "потому", "этого", "какой", "совсем", "ним",
    "здесь", "этом", "один", "почти", "мой", "тем", "чтобы", "нее", "сейчас",
    "были", "куда", "зачем", "всех", "никогда", "можно", "при", "наконец",
    "два", "об", "другой", "хоть", "после", "над", "больше", "тот", "через",
    "эти", "нас", "про", "всего", "них", "какая", "много", "разве", "три",
    "эту", "моя", "впрочем", "хорошо", "свою", "этой", "перед", "иногда",
    "лучше", "чуть", "том", "нельзя", "такой", "им", "более", "всегда",
    "конечно", "всю", "между", "также",
    # юридический шум, ставший общеупотребительным
    "суд", "вс", "рф", "дело", "определение", "постановление",
}


# Маркеры структурных секций обзора (упорядочены по приоритету)
SECTION_MARKERS = {
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


def parse_sections(text: str) -> dict:
    """
    Разбирает текст обзора на структурные секции.

    Возвращает словарь с ключами: fabula, lower_courts, vs_position, residual.
    Если маркеры секций отсутствуют, текст идёт целиком в residual.
    """
    sections = {"fabula": "", "lower_courts": "", "vs_position": "", "residual": ""}

    # Находим позиции всех маркеров в тексте
    found = []  # (start_pos, end_pos, section_name)
    for sec_name, patterns in SECTION_MARKERS.items():
        for p in patterns:
            for m in re.finditer(p, text, re.IGNORECASE):
                found.append((m.start(), m.end(), sec_name))

    if not found:
        sections["residual"] = text
        return sections

    # Сортируем по позиции
    found.sort()
    # Склеиваем секции: каждая секция = от конца её маркера до начала следующего маркера
    for i, (_, end_pos, sec_name) in enumerate(found):
        next_start = found[i + 1][0] if i + 1 < len(found) else len(text)
        chunk = text[end_pos:next_start].strip()
        # Если уже есть текст в этой секции (несколько маркеров), добавляем
        if sections[sec_name]:
            sections[sec_name] += "\n" + chunk
        else:
            sections[sec_name] = chunk

    # Если есть текст до первого маркера — это остаток (обычно заголовок и реквизиты)
    if found[0][0] > 0:
        sections["residual"] = text[: found[0][0]].strip()

    return sections


def normalize_case_id(case_number: str) -> str:
    """
    Извлекает канонический идентификатор для дедупликации.

    Приоритеты:
      1. Номер кассационной жалобы вида 305-ЭС24-12345 (наиболее уникальный).
      2. Номер арбитражного дела вида А40-12345/2023.
      3. Номер гражданского дела вида 5-КГ20-1234.
      4. Просто номер определения как fallback.
    """
    if not case_number:
        return ""

    # Очищаем от переносов строк и лишних пробелов
    text = re.sub(r"\s+", " ", case_number).strip()

    # 1. Номер ВС вида XXX-ЭСXX-XXXXX или XXX-КГXX-XXXXX
    vs_match = re.search(r"\b(\d{2,3})[\s-]?[ЭЭКк][СсГг][\s-]?(\d{1,2})[\s-]?(\d{2,6})", text)
    if vs_match:
        return f"{vs_match.group(1)}-{vs_match.group(2)}-{vs_match.group(3)}"

    # 2. Арбитражный номер вида А40-12345/2023
    arb_match = re.search(r"\bА\s*(\d{1,3})[\s-](\d{1,7})\s*/\s*(\d{4})", text)
    if arb_match:
        return f"А{arb_match.group(1)}-{arb_match.group(2)}/{arb_match.group(3)}"

    # 3. Гражданский номер вида 5-КГ20-1234
    civ_match = re.search(r"\b(\d{1,3})[\s-]?КГ[\s-]?(\d{1,2})[\s-]?(\d{1,6})", text)
    if civ_match:
        return f"{civ_match.group(1)}-КГ{civ_match.group(2)}-{civ_match.group(3)}"

    return ""


def parse_date(date_str: str) -> str:
    """Преобразует ДД.ММ.ГГГГ в YYYY-MM-DD. Возвращает '' если не парсится."""
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


class Lemmatizer:
    """Кэширующий лемматизатор. На корпусе из 5300 обзоров кэш реально ускоряет."""

    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()
        self.cache: dict = {}

    def tokenize(self, text: str) -> list:
        """Возвращает список лемм из текста, без стоп-слов."""
        if not text:
            return []
        tokens = re.findall(r"[а-яёa-z0-9]+", text.lower())
        result = []
        for t in tokens:
            if len(t) < 2:
                continue
            if t in STOPWORDS:
                continue
            if t in self.cache:
                lemma = self.cache[t]
            else:
                lemma = self.morph.parse(t)[0].normal_form
                self.cache[t] = lemma
            if lemma in STOPWORDS:
                continue
            result.append(lemma)
        return result


def build_index(db_path: Path, out_path: Path, meta_path: Path):
    print(f"Загружаю базу из {db_path}...")
    with open(db_path, encoding="utf-8") as f:
        db = json.load(f)
    print(f"  {len(db)} обзоров")

    print("Инициализирую лемматизатор...")
    lem = Lemmatizer()

    print("Обогащаю записи (парсинг секций, лемматизация, нормализация)...")
    enriched = []
    section_stats = {"fabula": 0, "lower_courts": 0, "vs_position": 0, "residual_only": 0}

    for i, msg in enumerate(db):
        if i % 500 == 0 and i > 0:
            print(f"  {i}/{len(db)}...")

        sections = parse_sections(msg.get("text", ""))

        # Статистика
        if sections["vs_position"]:
            section_stats["vs_position"] += 1
        if sections["fabula"]:
            section_stats["fabula"] += 1
        if sections["lower_courts"]:
            section_stats["lower_courts"] += 1
        if sections["residual"] and not (sections["fabula"] or sections["vs_position"]):
            section_stats["residual_only"] += 1

        # Лемматизация секций
        title = msg.get("title", "")
        title_lemmas = lem.tokenize(title)
        fabula_lemmas = lem.tokenize(sections["fabula"])
        lower_lemmas = lem.tokenize(sections["lower_courts"])
        vs_lemmas = lem.tokenize(sections["vs_position"])
        residual_lemmas = lem.tokenize(sections["residual"])

        # Если структуры нет, residual = весь текст. Считаем его как "vs_position",
        # чтобы не терять обзоры без маркеров при поиске по позиции ВС.
        if not vs_lemmas and not fabula_lemmas:
            vs_lemmas = residual_lemmas

        # Полный текст для гибридного скоринга
        full_lemmas = title_lemmas + fabula_lemmas + lower_lemmas + vs_lemmas + residual_lemmas

        # Хештеги — отдельный канал (для точного совпадения тегов)
        tag_lemmas = []
        for tag in msg.get("hashtags", []):
            tag_lemmas.extend(lem.tokenize(tag))

        # Нормализованный case_id и год
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

    print()
    print("Статистика парсинга секций:")
    for k, v in section_stats.items():
        pct = 100 * v / len(db)
        print(f"  {k:20s}: {v:5d} ({pct:5.1f}%)")

    # Группировка дубликатов по case_id
    print()
    print("Группирую дубликаты по case_id...")
    case_groups: dict = {}
    for doc in enriched:
        cid = doc["case_id"]
        if cid:
            case_groups.setdefault(cid, []).append(doc["id"])
    dup_count = sum(1 for ids in case_groups.values() if len(ids) > 1)
    total_in_dup = sum(len(ids) for ids in case_groups.values() if len(ids) > 1)
    print(f"  Уникальных дел: {len(case_groups)}")
    print(f"  Дел с несколькими обзорами: {dup_count}")
    print(f"  Записей в группах с дубликатами: {total_in_dup}")

    # Строим BM25-индексы по секциям
    print()
    print("Строю BM25-индексы по секциям...")
    fields = ["title", "fabula", "vs_position", "full", "tags"]
    bm25_indexes = {}
    for field in fields:
        corpus = [doc["lemmas"][field] if doc["lemmas"][field] else [""] for doc in enriched]
        bm25_indexes[field] = BM25Okapi(corpus)
        print(f"  {field:15s}: построен")

    # Кэш лемматизатора сохраняем — пригодится в search.py для типичных юр-терминов
    print(f"\nРазмер кэша лемм: {len(lem.cache)}")

    # Сериализация
    index_data = {
        "version": 1,
        "built_at": datetime.now().isoformat(),
        "documents": enriched,
        "bm25": bm25_indexes,
        "case_groups": case_groups,
        "lemma_cache": lem.cache,
    }

    # Сериализация: пишем gzip-сжатый pickle для компактности (~11 МБ vs 54 МБ raw).
    # При желании можно сохранить и обычный pkl через --no-gzip.
    if str(out_path).endswith(".gz"):
        import gzip
        print(f"\nСохраняю индекс в {out_path} (gzip)...")
        with gzip.open(out_path, "wb", compresslevel=9) as f:
            pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print(f"\nСохраняю индекс в {out_path}...")
        with open(out_path, "wb") as f:
            pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  Размер: {size_mb:.1f} МБ")

    # Метаинформация в JSON для диагностики
    meta = {
        "version": 1,
        "built_at": index_data["built_at"],
        "documents_count": len(enriched),
        "section_stats": section_stats,
        "duplicates": {
            "unique_cases": len(case_groups),
            "cases_with_dups": dup_count,
            "records_in_dups": total_in_dup,
        },
        "fields": fields,
        "lemma_cache_size": len(lem.cache),
        "index_size_mb": round(size_mb, 1),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Метаинформация сохранена в {meta_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=None, help="Путь к court_practice_db.json")
    parser.add_argument("--out", default=None, help="Путь к выходному index.pkl")
    parser.add_argument("--meta", default=None, help="Путь к выходному index_meta.json")
    parser.add_argument("--rebuild", action="store_true", help="Переиндексировать даже если индекс существует")
    args = parser.parse_args()

    here = Path(__file__).parent.parent
    db_path = Path(args.db) if args.db else here / "data" / "court_practice_db.json"
    out_path = Path(args.out) if args.out else here / "data" / "index.pkl.gz"
    meta_path = Path(args.meta) if args.meta else here / "data" / "index_meta.json"

    if out_path.exists() and not args.rebuild:
        print(f"Индекс уже существует: {out_path}")
        print("Используйте --rebuild для переиндексации")
        sys.exit(0)

    build_index(db_path, out_path, meta_path)


if __name__ == "__main__":
    main()
