"""Paired bootstrap over projects for micro/macro-F1 differences, plus Holm correction.

Rows of the two systems must be aligned by project (pair_id): resampling a row resamples
the same project for both, so EN/TR/code-switch comparisons stay paired.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence

import numpy as np

SEED = 42  # default random seed of every bootstrap here; result files record it


def resample_counts(rows: int, resamples: int = 1000, seed: int = SEED) -> np.ndarray:
    """How often each row is drawn in each bootstrap replicate (each row sums to ``rows``)."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(rows, np.full(rows, 1 / rows), size=resamples).astype(np.float32)


def f1_scores(
    weights: np.ndarray, truth: np.ndarray, predicted: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Micro- and macro-F1 for every row of ``weights`` (zero-division counts as 0)."""
    truth, predicted = truth.astype(bool), predicted.astype(bool)
    tp = weights @ (truth & predicted).astype(np.float32)
    errors = weights @ (truth ^ predicted).astype(np.float32)
    micro = _ratio(2 * tp.sum(axis=1), 2 * tp.sum(axis=1) + errors.sum(axis=1))
    macro = _ratio(2 * tp, 2 * tp + errors).mean(axis=1)
    return micro, macro


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0
    )


def paired_bootstrap(
    truth_a: np.ndarray,
    predicted_a: np.ndarray,
    truth_b: np.ndarray,
    predicted_b: np.ndarray,
    resamples: int = 1000,
    seed: int = SEED,
    groups: Sequence[Hashable] | np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """Difference A - B with a 95% percentile interval and a two-sided bootstrap p-value.

    With ``groups`` (one project id per row, e.g. pair_id) whole projects are resampled and
    every variant row of a project carries its project's multiplicity. The p-value uses the
    add-one finite-resample correction, so it never drops below ``p_resolution``.
    """
    if truth_a.shape != truth_b.shape or predicted_a.shape != predicted_b.shape:
        raise ValueError("Systems must be paired: same projects in the same row order.")
    rows = len(truth_a)
    if groups is not None and len(groups) != rows:
        raise ValueError(f"Need one group id per row: {len(groups)} groups for {rows} rows.")
    unit = np.ones((1, rows), dtype=np.float32)
    if groups is None:
        weights = resample_counts(rows, resamples, seed)
    else:
        unique, member = np.unique(np.asarray(groups), return_inverse=True)
        weights = resample_counts(len(unique), resamples, seed)[:, member.ravel()]
    point_a, point_b = f1_scores(unit, truth_a, predicted_a), f1_scores(unit, truth_b, predicted_b)
    boot_a, boot_b = f1_scores(weights, truth_a, predicted_a), f1_scores(
        weights, truth_b, predicted_b
    )
    result = {}
    for index, metric in enumerate(("micro_f1", "macro_f1")):
        differences = boot_a[index] - boot_b[index]
        low, high = np.percentile(differences, [2.5, 97.5])
        tail = min(np.sum(differences <= 0), np.sum(differences >= 0))
        result[metric] = {
            "a": float(point_a[index][0]),
            "b": float(point_b[index][0]),
            "difference": float(point_a[index][0] - point_b[index][0]),
            "ci_low": float(low),
            "ci_high": float(high),
            "p_value": float(min(1.0, 2 * (1 + tail) / (1 + resamples))),
            "p_resolution": 2 / (1 + resamples),
        }
    return result


def holm(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (step-down, kept monotone, capped at 1)."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted, running = {}, 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * value))
        adjusted[name] = running
    return adjusted
