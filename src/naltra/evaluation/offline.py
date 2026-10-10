"""Threshold tuning and ensemble voting over saved (members x documents x labels) scores."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from naltra.evaluation.matrix import flat_metrics

THRESHOLD_GRID = np.round(np.arange(0.02, 0.99, 0.02), 2)


def split_halves(pair_ids: Sequence[str]) -> np.ndarray:
    """Deterministic validation half per pair_id: False = val-A, True = val-B."""
    return np.array(
        [
            int(hashlib.sha256(pair.encode("utf-8")).hexdigest()[:8], 16) % 2 == 1
            for pair in pair_ids
        ]
    )


def tune_threshold(
    truth: np.ndarray, scores: np.ndarray, grid: np.ndarray = THRESHOLD_GRID
) -> tuple[float, float]:
    """Return the global threshold with the best micro-F1 and that F1 (ties: lowest)."""
    best = max(((flat_metrics(truth, scores >= t)["micro_f1"], -t) for t in grid), key=lambda x: x)
    return float(-best[1]), float(best[0])


def hard_vote(selections: np.ndarray) -> np.ndarray:
    """Fraction of members that selected each (document, label)."""
    return selections.astype(float).mean(axis=0)


def strict_majority(members: int) -> float:
    """Smallest vote fraction that is strictly more than half of the members."""
    return (members // 2 + 1) / members


def tune_k(truth: np.ndarray, fraction: np.ndarray, members: int) -> tuple[int, float]:
    """Best k-of-M vote count by micro-F1 (ties: the larger, more precise k)."""
    scored = [
        (flat_metrics(truth, fraction >= k / members - 1e-9)["micro_f1"], k)
        for k in range(1, members + 1)
    ]
    f1, k = max(scored)
    return k, float(f1)


def weighted_soft_vote(scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean of member probabilities."""
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (scores.shape[0],) or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Weights must be nonnegative, one per member, with a positive sum.")
    return np.tensordot(weights / weights.sum(), scores, axes=1)
