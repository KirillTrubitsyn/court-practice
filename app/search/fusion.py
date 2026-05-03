"""Reciprocal Rank Fusion. Веса слоёв и параметр k вынесены в конфиг."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]],
    weights: Sequence[float] | None = None,
    k: int = 60,
) -> list[tuple[int, float]]:
    """Слить несколько ранжированных списков id в общий рейтинг.

    Args:
        rankings: список ранжированных списков doc_id; первый элемент — лучший.
        weights: вес каждого ранжирования; по умолчанию все = 1.
        k: сглаживающая константа RRF.

    Returns:
        Список (doc_id, score), отсортированный по убыванию score.
    """
    if not rankings:
        return []

    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("len(weights) должен совпадать с len(rankings)")

    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        if weight <= 0:
            continue
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank + 1)

    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def topn(
    fused: Iterable[tuple[int, float]],
    n: int,
) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for i, item in enumerate(fused):
        if i >= n:
            break
        out.append(item)
    return out
