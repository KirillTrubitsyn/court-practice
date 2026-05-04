"""Sanity-check года определения по case_number и якорю в тексте обзора.

Парсер дат в исходных Telegram-каналах часто подцепляет первую попавшуюся
DD.MM.YYYY в тексте (дату конвенции, закона, нижестоящего суда). Это даёт
артефакты вроде date=09.09.1886 при case_number=А50-29022/2018.

Стратегия (по убыванию надёжности):
1. Якорь "Определение/Постановление ... от ДД.ММ.ГГГГ" в начале текста.
2. Год из case_number (ЭС/КГ format = год регистрации в ВС, надёжный).
3. Год из case_number (А-формат = год начала спора, менее надёжный).
4. Fallback на исходный date_year.
"""

from __future__ import annotations

import re


# Якорь "Определение от ДД.ММ.ГГГГ" — самый надёжный источник.
# Допускаем «Определение Верховного Суда РФ от …», «Определением СКЭС ВС РФ от …».
_ANCHOR_DATE = re.compile(
    r"[ОоОО]пределени[ея][а-яё\s.,№-]{0,80}?\bот\s+(\d{1,2})\.(\d{1,2})\.(\d{4})\b"
)

# ВС-кассация: "305-ЭС24-12345" → 24 (год регистрации в ВС).
_VS_YEAR = re.compile(r"\b\d{2,3}[\s-]?[Ээ][Сс][\s-]?(\d{2})\b")
# Гражданская кассация: "5-КГ22-300" → 22.
_KG_YEAR = re.compile(r"\b\d{1,3}[\s-]?КГ[\s-]?(\d{2})\b")
# Уголовная кассация: "225-УД24-1-А6", "51-УД25-4-К8" → 24/25.
_UD_YEAR = re.compile(r"\b\d{1,4}[\s-]?УД[\s-]?(\d{2})\b")
# Административная кассация: "А-АПЛ22-...".
_APL_YEAR = re.compile(r"\bАПЛ[\s-]?(\d{2})\b")
# Арбитражный: "А40-12345/2021" → 2021 (год начала спора, ВС обычно решал позже).
_ARB_YEAR = re.compile(r"/(\d{4})")


def _two_digit_to_year(yy: int) -> int:
    """`24` → 2024, `99` → 1999. Граница 50: всё <50 это 20XX, >=50 это 19XX."""
    return 2000 + yy if yy < 50 else 1900 + yy


def year_from_anchor(text: str, max_chars: int = 800) -> int | None:
    """Ищем "Определение ... от ДД.ММ.ГГГГ" в начале текста."""
    if not text:
        return None
    head = text[:max_chars]
    m = _ANCHOR_DATE.search(head)
    if m:
        try:
            year = int(m.group(3))
            if 2010 <= year <= 2099:
                return year
        except ValueError:
            pass
    return None


def year_from_case_number(case_number: str) -> int | None:
    """Извлечь год регистрации дела из реквизита. None если не парсится.

    Возвращаем год регистрации в ВС (ЭС/КГ) если есть, иначе арбитражный год.
    Год из ЭС/КГ — это когда ВС взял дело, обычно близко к дате определения.
    """
    if not case_number:
        return None
    text = re.sub(r"\s+", " ", case_number)

    m = _VS_YEAR.search(text)
    if m:
        return _two_digit_to_year(int(m.group(1)))

    m = _KG_YEAR.search(text)
    if m:
        return _two_digit_to_year(int(m.group(1)))

    m = _UD_YEAR.search(text)
    if m:
        return _two_digit_to_year(int(m.group(1)))

    m = _APL_YEAR.search(text)
    if m:
        return _two_digit_to_year(int(m.group(1)))

    m = _ARB_YEAR.search(text)
    if m:
        try:
            year = int(m.group(1))
            if 1990 <= year <= 2099:
                return year
        except ValueError:
            pass

    return None


def reconcile_year(
    date_year: int | None,
    anchor_year: int | None,
    case_year: int | None,
) -> int | None:
    """Выбрать «правильный» год определения ВС.

    Правила:
    - case_year — нижняя граница реальности: определение ВС не может быть
      раньше года регистрации дела в ВС. Если он известен — игнорируем
      любой anchor/date_year < case_year (это цитированные старые акты).
    - Из «допустимых» (≥ case_year) кандидатов выбираем максимум: anchor
      обычно совпадает с реальной датой определения, но если в тексте
      есть и более новое — берём его.
    - Если все источники меньше case_year — fallback на case_year.
    - Если case_year неизвестен — приоритет anchor, потом date_year.
    """
    floor = case_year
    candidates: list[int] = []
    if anchor_year is not None and (floor is None or anchor_year >= floor):
        candidates.append(anchor_year)
    if date_year is not None and (floor is None or date_year >= floor):
        candidates.append(date_year)
    if candidates:
        return max(candidates)
    return floor
