#!/usr/bin/env python3
"""
Гибридный поиск по индексированной базе судебной практики ВС РФ.

Использование:
  python3 scripts/search.py "запрос"
  python3 scripts/search.py "запрос" --court СКЭС --limit 15
  python3 scripts/search.py "запрос" --tag банкротство
  python3 scripts/search.py "запрос" --article "ст. 333 ГК"
  python3 scripts/search.py "запрос" --year-from 2023
  python3 scripts/search.py "запрос" --full --limit 5
  python3 scripts/search.py "" --tag залог --limit 20
  python3 scripts/search.py --stats

Режимы:
  по умолчанию      — гибрид BM25 + семантика через RRF (если есть embeddings.npz)
  --no-semantic     — чистый BM25 (как было до Уровня 2)
  --semantic-only   — чистый семантический поиск (для диагностики)
  --explain         — разбор скоринга по компонентам

Алгоритм гибрида:
  1. BM25 по взвешенным секциям → top-50 кандидатов
  2. Cosine similarity к title и body эмбеддингам → top-50 кандидатов
     (берём max sim из двух)
  3. Reciprocal Rank Fusion: score = w_bm25/(60+rank_bm25) + w_sem/(60+rank_sem)
  4. Применяются фильтры (court, tag, year, article)
  5. Boost за свежесть и фразовое соседство (как в BM25)
  6. Дедупликация по case_id

Семантика делает погоду на запросах с синонимией («снятие корпоративной вуали»,
«виндикация»), BM25 — на точных терминах и редких словах.

Зависимости: pymorphy3, rank_bm25, voyageai (для эмбеддинга запроса), numpy.
Для семантики требуется VOYAGE_API_KEY в окружении.
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
    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")
    if missing:
        print(f"Устанавливаю зависимости: {', '.join(missing)}...", file=sys.stderr)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--break-system-packages", *missing],
            check=True,
        )


_ensure_deps()


import argparse
import hashlib
import json
import math
import os
import pickle
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pymorphy3


def _load_index_pickle(path: Path):
    """Загружает индекс из .pkl или .pkl.gz (по сигнатуре gzip-magic 0x1f 0x8b)."""
    with open(path, "rb") as f:
        magic = f.read(2)
        f.seek(0)
        if magic == b"\x1f\x8b":
            import gzip
            with gzip.open(f, "rb") as gz:
                return pickle.load(gz)
        return pickle.load(f)


def _load_voyage_key() -> str:
    """
    Загружает API-ключ Voyage в порядке приоритета:
      1. Переменная окружения VOYAGE_API_KEY (стандартный способ для CLI/dev)
      2. Файл data/.voyage_config.json в скилле (для claude.ai web, где env-переменные
         недоступны)
      3. Возвращает пустую строку — тогда работает только BM25 (graceful fallback)
    """
    env_key = os.environ.get("VOYAGE_API_KEY", "").strip()
    if env_key:
        return env_key

    config_path = Path(__file__).parent.parent / "data" / ".voyage_config.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
            key = cfg.get("voyage_api_key", "").strip()
            if key:
                return key
        except (OSError, json.JSONDecodeError):
            pass

    return ""


# Веса секций в BM25-скоринге (подобраны эмпирически)
SECTION_WEIGHTS = {
    "title": 3.0,         # заголовок концентрирует суть обзора
    "vs_position": 3.0,   # позиция ВС РФ — самое ценное
    "full": 1.0,          # общий контекст
    "fabula": 0.7,        # фабула — фон, важна меньше позиции
    "tags": 5.0,          # точное тематическое совпадение
}

# Параметры boost по свежести: мягкая экспонента, не обнуляющая старую практику
FRESHNESS_BOOST_MAX = 0.30  # +30% максимум для самых свежих
FRESHNESS_HALFLIFE_YEARS = 4.0

# Гибридный поиск: параметры RRF (Reciprocal Rank Fusion)
RRF_K = 60                 # классическая константа из статьи Cormack (2009)
HYBRID_BM25_WEIGHT = 1.0   # вес BM25 в гибриде
HYBRID_SEM_WEIGHT = 1.0    # вес семантики в гибриде
HYBRID_TOP_K = 100         # сколько кандидатов берём от каждого слоя для RRF
SEM_BODY_WEIGHT = 0.7      # как взвешивать body-эмбеддинг относительно title

# Те же стоп-слова, что в index.py
STOPWORDS = {
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
    "суд", "вс", "рф", "дело", "определение", "постановление",
}


class Searcher:
    def __init__(self, index_path: Path, embeddings_path: Path = None,
                 query_cache_path: Path = None):
        self.idx = _load_index_pickle(index_path)
        self.docs = self.idx["documents"]
        self.bm25 = self.idx["bm25"]
        self.case_groups = self.idx["case_groups"]
        self.morph = pymorphy3.MorphAnalyzer()
        # Стартуем с предзагруженным кэшем из индексации
        self.lemma_cache: dict = dict(self.idx.get("lemma_cache", {}))

        # Загрузка эмбеддингов (необязательно — без них работаем как чистый BM25)
        self.doc_emb = None
        if embeddings_path and embeddings_path.exists():
            try:
                arr = np.load(embeddings_path, allow_pickle=False)
                self.doc_emb = arr.astype(np.float32)
                if len(self.doc_emb) != len(self.docs):
                    print(f"WARNING: размер эмбеддингов ({len(self.doc_emb)}) "
                          f"не совпадает с числом документов ({len(self.docs)}). "
                          f"Семантика отключена.", file=sys.stderr)
                    self.doc_emb = None
            except Exception as e:
                print(f"WARNING: не удалось загрузить эмбеддинги: {e}", file=sys.stderr)

        # Кэш эмбеддингов запросов на диске (чтобы не платить за повторные)
        self.query_cache_path = query_cache_path
        self.query_cache: dict = {}
        if query_cache_path and query_cache_path.exists():
            try:
                with open(query_cache_path, "rb") as f:
                    self.query_cache = pickle.load(f)
            except Exception:
                self.query_cache = {}

    def has_semantic(self) -> bool:
        return self.doc_emb is not None

    def _embed_query(self, query: str) -> np.ndarray:
        """Эмбеддит запрос через Voyage. Возвращает нормализованный np.float32 (1024,).
        Использует диск-кэш, чтобы не повторять API-вызовы за один и тот же текст.
        """
        norm_q = re.sub(r"\s+", " ", query.strip().lower())
        cache_key = hashlib.sha256(norm_q.encode("utf-8")).hexdigest()[:16]

        if cache_key in self.query_cache:
            return self.query_cache[cache_key]

        api_key = _load_voyage_key()
        if not api_key:
            raise RuntimeError(
                "VOYAGE_API_KEY не найден. Семантика недоступна. "
                "Положите ключ в env-переменную VOYAGE_API_KEY или в файл "
                "data/.voyage_config.json. Или используйте --no-semantic для чистого BM25."
            )

        try:
            import voyageai
        except ImportError:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", "--break-system-packages", "voyageai"],
                check=True,
            )
            import voyageai

        client = voyageai.Client(api_key=api_key)
        result = client.embed(
            texts=[query], model="voyage-3-large", input_type="query",
        )
        vec = np.array(result.embeddings[0], dtype=np.float32)
        # Нормализуем для cosine = dot product
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        self.query_cache[cache_key] = vec
        if self.query_cache_path:
            try:
                with open(self.query_cache_path, "wb") as f:
                    pickle.dump(self.query_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print(f"WARNING: не удалось сохранить кэш запросов: {e}", file=sys.stderr)
        return vec

    def _semantic_scores(self, query: str) -> np.ndarray:
        """Cosine similarity для каждого документа. Возвращает массив длины N."""
        if not self.has_semantic():
            return np.zeros(len(self.docs), dtype=np.float32)
        q_vec = self._embed_query(query)
        # doc_emb уже нормализован при индексации, поэтому dot product = cosine
        return self.doc_emb @ q_vec  # (N,)

    def lemmatize_query(self, query: str) -> list:
        if not query:
            return []
        tokens = re.findall(r"[а-яёa-z0-9]+", query.lower())
        result = []
        for t in tokens:
            if len(t) < 2 or t in STOPWORDS:
                continue
            if t in self.lemma_cache:
                lemma = self.lemma_cache[t]
            else:
                lemma = self.morph.parse(t)[0].normal_form
                self.lemma_cache[t] = lemma
            if lemma in STOPWORDS:
                continue
            result.append(lemma)
        return result

    def freshness_boost(self, year: int) -> float:
        """Лёгкая поправка по году. Самые свежие до +30%, старая практика не штрафуется."""
        if not year:
            return 1.0
        current_year = datetime.now().year
        years_ago = max(0, current_year - year)
        boost = FRESHNESS_BOOST_MAX * math.exp(-years_ago / FRESHNESS_HALFLIFE_YEARS)
        return 1.0 + boost

    def passes_filters(
        self,
        doc: dict,
        court_filter: str = None,
        tag_filter: str = None,
        article_filter: str = None,
        year_from: int = None,
        year_to: int = None,
    ) -> bool:
        if court_filter:
            cf = court_filter.upper()
            if cf == "СКГД" and "СКГД" not in doc["court"]:
                return False
            if cf == "СКЭС" and "СКЭС" not in doc["court"]:
                return False

        if tag_filter:
            tf = tag_filter.lower()
            if not any(t.lower() == tf for t in doc["hashtags"]):
                return False

        if article_filter:
            af_norm = article_filter.lower()
            text_lower = doc["text"].lower()
            if af_norm not in text_lower:
                # Гибкое соответствие: ищем все числа из фильтра
                nums = re.findall(r"\d+", article_filter)
                if not nums or not all(n in text_lower for n in nums):
                    return False

        if year_from and (not doc["year"] or doc["year"] < year_from):
            return False
        if year_to and (not doc["year"] or doc["year"] > year_to):
            return False

        return True

    def _proximity_score(self, doc: dict, query_lemmas: list) -> float:
        """
        Скор фразового соседства: проверяем все секции документа на наличие
        пар лемм из запроса в окне ≤ 3 позиций. Возвращает [0..1]:
        0 — нигде не встретились рядом, 1 — все пары встретились в одной секции.
        """
        if len(query_lemmas) < 2:
            return 0.0

        # Все возможные пары соседних лемм запроса
        pairs = [(query_lemmas[i], query_lemmas[i + 1]) for i in range(len(query_lemmas) - 1)]

        best_score = 0.0
        for section in ("title", "vs_position", "fabula", "full"):
            lemmas = doc["lemmas"][section]
            if not lemmas:
                continue
            # Позиции каждой леммы запроса в секции
            positions = {ql: [] for ql in query_lemmas}
            for i, lem in enumerate(lemmas):
                if lem in positions:
                    positions[lem].append(i)

            matched_pairs = 0
            for a, b in pairs:
                # Есть ли позиции a и b в окне ≤ 3
                if not positions[a] or not positions[b]:
                    continue
                found = False
                for pa in positions[a]:
                    for pb in positions[b]:
                        if 0 < abs(pa - pb) <= 3:
                            found = True
                            break
                    if found:
                        break
                if found:
                    matched_pairs += 1

            if pairs:
                section_score = matched_pairs / len(pairs)
                # Совпадение в title или vs_position ценнее, чем в fabula
                if section in ("title", "vs_position"):
                    section_score *= 1.5
                best_score = max(best_score, section_score)

        return min(best_score, 1.0)

    def search(
        self,
        query: str,
        court_filter: str = None,
        tag_filter: str = None,
        article_filter: str = None,
        year_from: int = None,
        year_to: int = None,
        limit: int = 10,
        deduplicate: bool = True,
        explain: bool = False,
        mode: str = "hybrid",  # "hybrid", "bm25", "semantic"
    ) -> tuple:
        """Возвращает (results, total_found_before_limit).

        mode:
          "hybrid"   — BM25 + семантика через RRF (по умолчанию)
          "bm25"     — только BM25 (как до Уровня 2)
          "semantic" — только семантика (для диагностики)
        """
        query_lemmas = self.lemmatize_query(query)

        # Если режим требует семантику, но эмбеддингов нет — откатываемся
        if mode in ("hybrid", "semantic") and not self.has_semantic():
            mode = "bm25"

        # Если совсем нет запроса и нет фильтров — пусто
        if not query_lemmas and not (tag_filter or article_filter or court_filter):
            return [], 0

        # ---- Шаг 1: фильтрация (применяется во всех режимах) ----
        eligible_indices = []
        for i, doc in enumerate(self.docs):
            if self.passes_filters(doc, court_filter, tag_filter, article_filter,
                                   year_from, year_to):
                eligible_indices.append(i)

        if not eligible_indices:
            return [], 0

        # ---- Шаг 2: BM25-скоринг (если есть запрос и режим использует BM25) ----
        bm25_scores = {}  # doc_index -> total BM25 score
        if query_lemmas and mode in ("bm25", "hybrid"):
            section_scores = {}
            for field in SECTION_WEIGHTS:
                section_scores[field] = self.bm25[field].get_scores(query_lemmas)

            for i in eligible_indices:
                total = 0.0
                for field, weight in SECTION_WEIGHTS.items():
                    total += section_scores[field][i] * weight
                if total > 0:
                    bm25_scores[i] = total

        # ---- Шаг 3: Семантический скоринг ----
        sem_scores = {}  # doc_index -> cosine similarity
        if query and mode in ("semantic", "hybrid"):
            try:
                sem_array = self._semantic_scores(query)
                for i in eligible_indices:
                    if sem_array[i] > 0:  # отбрасываем явный шум
                        sem_scores[i] = float(sem_array[i])
            except RuntimeError as e:
                if mode == "hybrid":
                    # В гибриде откатываемся на чистый BM25 с предупреждением
                    print(f"WARNING: семантика недоступна ({e}). "
                          f"Откат на BM25.", file=sys.stderr)
                    mode = "bm25"
                else:
                    # В semantic-only пробрасываем ошибку наружу
                    raise

        # ---- Шаг 4: Объединение результатов ----
        scored = []  # (final_score, doc, breakdown)

        if mode == "bm25":
            # Только BM25
            for i, score in bm25_scores.items():
                doc = self.docs[i]
                breakdown = {"bm25": round(score, 2)} if explain else None
                scored.append((score, doc, breakdown, i))

        elif mode == "semantic":
            # Только семантика — score = cosine similarity (без boost)
            for i, score in sem_scores.items():
                doc = self.docs[i]
                # Маштабируем для сопоставимости с BM25-скорами в выводе
                final_score = score * 100
                breakdown = {"sem": round(score, 4)} if explain else None
                scored.append((final_score, doc, breakdown, i))

        else:
            # Hybrid через RRF
            # Берём top-K из каждого списка
            bm25_ranked = sorted(bm25_scores.items(), key=lambda x: -x[1])[:HYBRID_TOP_K]
            sem_ranked = sorted(sem_scores.items(), key=lambda x: -x[1])[:HYBRID_TOP_K]

            bm25_rank_map = {doc_idx: rank for rank, (doc_idx, _) in enumerate(bm25_ranked, 1)}
            sem_rank_map = {doc_idx: rank for rank, (doc_idx, _) in enumerate(sem_ranked, 1)}

            # Кандидаты — объединение top-K обоих списков
            candidates = set(bm25_rank_map) | set(sem_rank_map)

            for i in candidates:
                rrf = 0.0
                breakdown = {} if explain else None

                if i in bm25_rank_map:
                    bm25_contrib = HYBRID_BM25_WEIGHT / (RRF_K + bm25_rank_map[i])
                    rrf += bm25_contrib
                    if explain:
                        breakdown["bm25_rank"] = bm25_rank_map[i]
                        breakdown["bm25_score"] = round(bm25_scores[i], 2)

                if i in sem_rank_map:
                    sem_contrib = HYBRID_SEM_WEIGHT / (RRF_K + sem_rank_map[i])
                    rrf += sem_contrib
                    if explain:
                        breakdown["sem_rank"] = sem_rank_map[i]
                        breakdown["sem_sim"] = round(sem_scores[i], 4)

                # Маштабируем RRF в диапазон, сопоставимый с BM25 (визуально приятнее)
                final_score = rrf * 1000
                doc = self.docs[i]
                scored.append((final_score, doc, breakdown, i))

        # ---- Шаг 5: Boost-факторы ----
        if scored and query_lemmas:
            scored_with_boost = []
            for score, doc, breakdown, doc_idx in scored:
                # Boost: свежесть
                fresh = self.freshness_boost(doc["year"])
                score *= fresh

                # Boost: структурированная позиция ВС
                if doc["sections"]["vs_position"]:
                    score *= 1.10
                    if explain and breakdown is not None:
                        breakdown["structured_vs"] = "x1.10"

                # Boost: фразовое соседство (актуально только если есть BM25-сигнал)
                if len(query_lemmas) >= 2:
                    proximity = self._proximity_score(doc, query_lemmas)
                    if proximity > 0:
                        prox_mult = 1.0 + 0.5 * proximity
                        score *= prox_mult
                        if explain and breakdown is not None:
                            breakdown["proximity_x"] = round(prox_mult, 3)

                if explain and breakdown is not None:
                    breakdown["freshness_x"] = round(fresh, 3)
                    breakdown["final"] = round(score, 2)

                scored_with_boost.append((score, doc, breakdown))
            scored = scored_with_boost
        else:
            scored = [(s, d, b) for s, d, b, _ in scored]

        scored.sort(key=lambda x: -x[0])
        total_before_dedup = len(scored)

        # ---- Шаг 6: Дедупликация по case_id ----
        if deduplicate:
            seen_cases = {}
            deduped = []
            for score, doc, expl in scored:
                cid = doc["case_id"]
                if cid:
                    if cid in seen_cases:
                        seen_cases[cid]["alternative_channels"].append(doc["source_channel"])
                        continue
                    entry = {
                        "score": score,
                        "doc": doc,
                        "explain": expl,
                        "alternative_channels": [],
                    }
                    seen_cases[cid] = entry
                    deduped.append(entry)
                else:
                    deduped.append({
                        "score": score, "doc": doc, "explain": expl,
                        "alternative_channels": [],
                    })
            return deduped[:limit], len(deduped)
        else:
            return ([
                {"score": s, "doc": d, "explain": e, "alternative_channels": []}
                for s, d, e in scored[:limit]
            ], total_before_dedup)


def extract_position_snippet(doc: dict, max_length: int = 600) -> str:
    """Возвращает текст позиции ВС (или residual, если структуры нет)."""
    pos = doc["sections"]["vs_position"]
    if not pos:
        pos = doc["sections"]["residual"]

    if len(pos) <= max_length:
        return pos

    # Обрезаем по границе предложения
    cut = pos[:max_length].rfind(".")
    if cut > max_length // 2:
        return pos[: cut + 1]
    return pos[:max_length] + "..."


def format_result(entry: dict, rank: int, show_full: bool = False, explain: bool = False) -> str:
    doc = entry["doc"]
    out = []
    out.append("=" * 70)
    header = f"#{rank}  |  {doc['court']}  |  {doc['date']}  |  score={entry['score']:.2f}"
    out.append(header)
    out.append("=" * 70)
    out.append(f"📌 {doc['title']}")

    if doc.get("case_number"):
        cn_clean = re.sub(r"\s+", " ", doc["case_number"]).strip()
        out.append(f"📄 {cn_clean[:120]}")

    if doc.get("hashtags"):
        out.append(f"🏷️  {'  '.join('#' + t for t in doc['hashtags'])}")

    if doc.get("articles"):
        articles = [a[:80] for a in doc["articles"][:3]]
        out.append(f"⚖️  Нормы: {'; '.join(articles)}")

    if entry["alternative_channels"]:
        chs = list(set(entry["alternative_channels"]))
        out.append(f"📰 Также в каналах: {', '.join(chs)}")

    if explain and entry.get("explain"):
        items = [f"{k}={v}" for k, v in entry["explain"].items()]
        out.append(f"🔬 Вклад: {', '.join(items)}")

    if show_full:
        out.append("")
        out.append("-" * 50)
        out.append(doc["text"])
    else:
        snippet = extract_position_snippet(doc)
        if snippet:
            out.append("")
            out.append("📋 Позиция ВС РФ:")
            out.append(snippet)

    out.append("")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="Гибридный поиск по судебной практике ВС РФ")
    parser.add_argument("query", nargs="?", default="", help="Поисковый запрос")
    parser.add_argument("--court", choices=["СКГД", "СКЭС", "скгд", "скэс"])
    parser.add_argument("--tag", help="Фильтр по хештегу (без #)")
    parser.add_argument("--article", help='Фильтр по статье закона (например "ст. 333 ГК")')
    parser.add_argument("--year-from", type=int, help="С какого года")
    parser.add_argument("--year-to", type=int, help="По какой год")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--full", action="store_true", help="Полный текст обзоров")
    parser.add_argument("--no-dedup", action="store_true", help="Не дедуплицировать по делу")
    parser.add_argument("--explain", action="store_true", help="Показать разбор скоринга")
    parser.add_argument("--no-semantic", action="store_true",
                        help="Отключить семантику, использовать только BM25")
    parser.add_argument("--semantic-only", action="store_true",
                        help="Только семантика, без BM25 (для диагностики)")
    parser.add_argument("--stats", action="store_true", help="Статистика индекса")
    parser.add_argument("--list-tags", action="store_true", help="Список тегов")
    parser.add_argument("--index", default=None, help="Путь к index.pkl")
    parser.add_argument("--embeddings", default=None, help="Путь к embeddings.npy")
    args = parser.parse_args()

    here = Path(__file__).parent.parent
    if args.index:
        index_path = Path(args.index)
    else:
        gz = here / "data" / "index.pkl.gz"
        plain = here / "data" / "index.pkl"
        index_path = gz if gz.exists() else plain
    embeddings_path = Path(args.embeddings) if args.embeddings else here / "data" / "embeddings.npy"
    query_cache_path = here / "data" / "query_cache.pkl"

    if not index_path.exists():
        print(f"ERROR: Индекс не найден: {index_path}", file=sys.stderr)
        print("Сначала запустите: python3 scripts/index.py", file=sys.stderr)
        sys.exit(1)

    s = Searcher(index_path, embeddings_path, query_cache_path)

    if args.stats:
        print(f"📊 Индекс судебной практики ВС РФ")
        print(f"   Построен: {s.idx.get('built_at', '?')}")
        print(f"   Обзоров: {len(s.docs)}")
        скгд = sum(1 for d in s.docs if "СКГД" in d["court"])
        скэс = sum(1 for d in s.docs if "СКЭС" in d["court"])
        print(f"   СКГД: {скгд}, СКЭС: {скэс}, прочее: {len(s.docs) - скгд - скэс}")
        years = sorted({d["year"] for d in s.docs if d["year"]})
        if years:
            print(f"   Период: {years[0]}—{years[-1]}")
        with_vs = sum(1 for d in s.docs if d["sections"]["vs_position"])
        print(f"   Со структурированной позицией ВС: {with_vs} ({100*with_vs/len(s.docs):.0f}%)")
        print(f"   Уникальных дел: {len(s.case_groups)}")
        print(f"   Семантика: {'✓ доступна (voyage-3-large, 1024d)' if s.has_semantic() else '✗ недоступна (нет embeddings.npy)'}")
        if s.query_cache:
            print(f"   Кэш запросов: {len(s.query_cache)} записей")
        return

    if args.list_tags:
        tags = {}
        for d in s.docs:
            for t in d["hashtags"]:
                tags[t] = tags.get(t, 0) + 1
        print(f"📑 Хештеги ({len(tags)} шт.):")
        for tag, cnt in sorted(tags.items(), key=lambda x: -x[1]):
            print(f"  #{tag}: {cnt}")
        return

    if not args.query and not args.tag and not args.article and not args.court:
        parser.print_help()
        return

    # Определяем режим
    if args.semantic_only:
        mode = "semantic"
    elif args.no_semantic:
        mode = "bm25"
    else:
        mode = "hybrid"

    results, total = s.search(
        query=args.query,
        court_filter=args.court,
        tag_filter=args.tag,
        article_filter=args.article,
        year_from=args.year_from,
        year_to=args.year_to,
        limit=args.limit,
        deduplicate=not args.no_dedup,
        explain=args.explain,
        mode=mode,
    )

    actual_mode = mode
    if mode != "bm25" and not s.has_semantic():
        actual_mode = "bm25 (семантика недоступна)"

    print(f"\n🔍 Найдено: {total} (показано {len(results)}) | режим: {actual_mode}")
    if args.query:
        print(f"   Запрос: {args.query}")
        ql = s.lemmatize_query(args.query)
        print(f"   Леммы:  {' '.join(ql)}")
    if args.court:
        print(f"   Коллегия: {args.court.upper()}")
    if args.tag:
        print(f"   Тег: #{args.tag}")
    if args.article:
        print(f"   Статья: {args.article}")
    if args.year_from or args.year_to:
        print(f"   Период: {args.year_from or '*'}—{args.year_to or '*'}")
    print()

    for i, entry in enumerate(results, 1):
        print(format_result(entry, i, show_full=args.full, explain=args.explain))


if __name__ == "__main__":
    main()
