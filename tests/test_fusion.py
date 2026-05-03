"""Тесты RRF."""

from __future__ import annotations

from app.search.fusion import reciprocal_rank_fusion


def test_single_ranking_preserves_order() -> None:
    fused = reciprocal_rank_fusion([[5, 1, 3]], k=60)
    ids = [doc_id for doc_id, _ in fused]
    assert ids == [5, 1, 3]


def test_two_rankings_promote_intersection() -> None:
    rank_a = [1, 2, 3, 4]
    rank_b = [3, 1, 2, 5]
    fused = reciprocal_rank_fusion([rank_a, rank_b], k=60)
    ids = [doc_id for doc_id, _ in fused]
    # 1 встречается высоко в обоих → лидирует.
    assert ids[0] == 1
    # 5 — только в B → последний.
    assert ids[-1] in (5, 4)


def test_weights_zero_excludes_ranking() -> None:
    rank_a = [1, 2, 3]
    rank_b = [9, 8, 7]
    fused = reciprocal_rank_fusion([rank_a, rank_b], weights=[1.0, 0.0], k=60)
    ids = [doc_id for doc_id, _ in fused]
    assert ids == [1, 2, 3]


def test_empty_rankings() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_weights_length_mismatch() -> None:
    import pytest

    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[1], [2]], weights=[1.0])
