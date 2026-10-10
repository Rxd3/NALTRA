"""Threshold tuning and ensemble voting over saved score matrices."""

from __future__ import annotations

import numpy as np
import pytest

from naltra.evaluation.offline import (
    hard_vote,
    split_halves,
    strict_majority,
    tune_k,
    tune_threshold,
    weighted_soft_vote,
)


def test_split_halves_is_deterministic_and_keeps_pairs_together() -> None:
    pairs = [f"cordis:{i}" for i in range(200)] * 2
    first, second = split_halves(pairs), split_halves(pairs)
    assert first.tolist() == second.tolist()
    assert first[:200].tolist() == first[200:].tolist()
    assert 0.35 < first.mean() < 0.65


def test_tune_threshold_maximizes_micro_f1() -> None:
    truth = np.array([[1, 0], [0, 1], [1, 0]])
    scores = np.array([[0.30, 0.10], [0.05, 0.25], [0.35, 0.20]])
    threshold, f1 = tune_threshold(truth, scores, grid=np.array([0.1, 0.22, 0.5]))
    assert threshold == pytest.approx(0.22)
    assert f1 == pytest.approx(1.0)


def test_hard_vote_counts_member_selections() -> None:
    selections = np.array(
        [
            [[1, 0, 1]],
            [[1, 1, 0]],
            [[0, 1, 0]],
        ]
    )
    fraction = hard_vote(selections)
    assert np.allclose(fraction, [[2 / 3, 2 / 3, 1 / 3]])
    assert strict_majority(3) == pytest.approx(2 / 3)
    assert strict_majority(4) == pytest.approx(3 / 4)
    assert (fraction >= strict_majority(3)).astype(int).tolist() == [[1, 1, 0]]


def test_tune_k_prefers_the_best_vote_count() -> None:
    truth = np.array([[1, 1, 1]])
    fraction = np.array([[2 / 3, 1 / 3, 1 / 3]])
    k, f1 = tune_k(truth, fraction, members=3)
    assert k == 1
    assert f1 == pytest.approx(1.0)


def test_weighted_soft_vote_normalizes_weights() -> None:
    scores = np.array([[[0.2, 0.8]], [[0.6, 0.4]]])
    blended = weighted_soft_vote(scores, np.array([3.0, 1.0]))
    assert np.allclose(blended, [[0.3, 0.7]])
    with pytest.raises(ValueError):
        weighted_soft_vote(scores, np.array([0.0, 0.0]))
