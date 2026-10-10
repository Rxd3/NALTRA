"""Numbers behind the paper figures, kept separate from plotting so they are testable."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from naltra.evaluation.matrix import flat_metrics


def _depths(nodes: Sequence[str], parents: Mapping[str, str | None]) -> np.ndarray:
    def depth(node: str | None) -> int:
        return 0 if node is None else 1 + depth(parents[node])

    return np.array([depth(node) for node in nodes])


def root_confusion(
    truth_closed: np.ndarray,
    predicted_closed: np.ndarray,
    nodes: Sequence[str],
    parents: Mapping[str, str | None],
) -> tuple[list[str], np.ndarray]:
    """Root errors for documents of gold root i, rows normalised by gold-root support.

    The diagonal is per-root recall, P(i predicted | i gold). Off-diagonal cell (i, j) is the
    false-root rate P(j predicted and j not gold | i gold), so a correctly predicted second
    gold root is not counted (rows need not sum to 1 because documents are multi-root).
    """
    columns = [index for index, node in enumerate(nodes) if parents[node] is None]
    gold = truth_closed[:, columns].astype(float)
    predicted = predicted_closed[:, columns].astype(float)
    counts = gold.T @ (predicted * (1 - gold)) + np.diag((gold * predicted).sum(axis=0))
    support = gold.sum(axis=0)[:, None]
    matrix = np.divide(counts, support, out=np.zeros_like(counts), where=support > 0)
    return [nodes[index] for index in columns], matrix


def depth_f1(
    truth_closed: np.ndarray,
    predicted_closed: np.ndarray,
    nodes: Sequence[str],
    parents: Mapping[str, str | None],
) -> dict[int, float]:
    """Micro-F1 restricted to the nodes at each depth of the closed label space."""
    depths = _depths(nodes, parents)
    return {
        int(level): flat_metrics(
            truth_closed[:, depths == level], predicted_closed[:, depths == level]
        )["micro_f1"]
        for level in np.unique(depths)
    }


def reliability_bins(
    truth: np.ndarray, scores: np.ndarray, bins: int = 10
) -> dict[str, np.ndarray]:
    """Top-1 reliability diagram data: mean confidence and accuracy per non-empty bin."""
    rows = np.arange(len(scores))
    top = scores.argmax(axis=1)
    confidence = scores[rows, top].astype(float)
    correct = truth[rows, top].astype(float)
    index = np.minimum((confidence * bins).astype(int), bins - 1)
    used = np.unique(index)
    return {
        "confidence": np.array([confidence[index == b].mean() for b in used]),
        "accuracy": np.array([correct[index == b].mean() for b in used]),
        "count": np.array([(index == b).sum() for b in used]),
    }
