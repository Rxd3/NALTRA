"""Vectorized paper metrics over (documents x labels) truth, prediction and score matrices.

Macro averages always span every column passed in, so callers must pass the full label
universe (473 direct labels or 586 closed nodes), never only the labels that occurred.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve


def binarize(label_sets: Iterable[Iterable[str]], labels: Sequence[str]) -> np.ndarray:
    """Return a uint8 indicator matrix whose columns follow ``labels``."""
    index = {label: column for column, label in enumerate(labels)}
    rows = list(label_sets)
    matrix = np.zeros((len(rows), len(labels)), dtype=np.uint8)
    for row, label_set in enumerate(rows):
        for label in label_set:
            if label not in index:
                raise ValueError(f"Label {label!r} is unknown to the label universe.")
            matrix[row, index[label]] = 1
    return matrix


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0
    )


def flat_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Micro/macro P/R/F1 plus document-averaged Jaccard and F1."""
    truth, predicted = truth.astype(bool), predicted.astype(bool)
    result = {}
    for average in ("micro", "macro"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, predicted, average=average, zero_division=0
        )
        result |= {
            f"{average}_precision": float(precision),
            f"{average}_recall": float(recall),
            f"{average}_f1": float(f1),
        }
    overlap = (truth & predicted).sum(axis=1)
    union = (truth | predicted).sum(axis=1)
    sizes = truth.sum(axis=1) + predicted.sum(axis=1)
    result["jaccard"] = float(_ratio(overlap, union).mean())
    result["document_f1"] = float(_ratio(2 * overlap, sizes).mean())
    result["mean_predicted_labels"] = float(predicted.sum(axis=1).mean())
    return result


def label_distribution(truth: np.ndarray) -> dict[str, float | int | None]:
    """Label imbalance of a gold matrix: why per-label accuracy is no benchmark metric here.

    ``all_negative_accuracy`` is the per-label accuracy of predicting no label at all.
    """
    if truth.ndim != 2 or 0 in truth.shape:
        raise ValueError("label_distribution needs at least one record and one label.")
    truth = truth.astype(bool)
    counts = truth.sum(axis=0)
    prevalence = counts / len(truth)
    seen = prevalence[counts > 0]
    top = np.sort(counts)[::-1][: int(np.ceil(0.1 * len(counts)))]
    rare = int((prevalence < 0.01).sum())
    return {
        "n_records": len(truth),
        "n_labels": len(counts),
        "mean_labels_per_record": float(truth.sum(axis=1).mean()),
        "median_labels_per_record": float(np.median(truth.sum(axis=1))),
        "min_prevalence": float(prevalence.min()),
        "median_prevalence": float(np.median(prevalence)),
        "max_prevalence": float(prevalence.max()),
        "rare_labels": rare,
        "rare_label_share": rare / len(counts),
        "top_decile_positive_share": float(top.sum() / counts.sum()) if counts.sum() else 0.0,
        "prevalence_ratio": float(seen.max() / seen.min()) if len(seen) else None,
        "zero_positive_labels": int((counts == 0).sum()),
        "all_negative_accuracy": float(1 - prevalence.mean()),
    }


def ranking_metrics(
    truth: np.ndarray, scores: np.ndarray, ks: Sequence[int] = (1, 3, 5)
) -> dict[str, float]:
    """Threshold-free P@k, R-precision and nDCG@5."""
    order = np.argsort(-scores, axis=1, kind="stable")
    hits = np.take_along_axis(truth.astype(bool), order, axis=1)
    result = {f"p_at_{k}": float(hits[:, :k].mean()) for k in ks}
    positives = truth.sum(axis=1)
    ranks = np.arange(hits.shape[1])
    within = ranks[None, :] < positives[:, None]
    result["r_precision"] = float(_ratio((hits & within).sum(axis=1), positives).mean())
    discounts = 1 / np.log2(np.arange(2, 7))
    depth = min(5, hits.shape[1])
    dcg = (hits[:, :depth] * discounts[:depth]).sum(axis=1)
    ideal = np.array([discounts[: min(int(p), depth)].sum() for p in positives])
    result["ndcg_at_5"] = float(_ratio(dcg, ideal).mean())
    return result


def calibration_metrics(truth: np.ndarray, scores: np.ndarray, bins: int = 10) -> dict[str, float]:
    """Pooled ECE/Brier over every document-label cell, plus top-1 ECE."""
    if bins < 1:
        raise ValueError("bins must be positive.")
    outcomes, confidences = truth.astype(float).ravel(), scores.astype(float).ravel()
    top = scores.argmax(axis=1)
    top_correct = truth[np.arange(len(truth)), top].astype(float)
    top_confidence = scores[np.arange(len(scores)), top].astype(float)
    return {
        "ece": _ece(confidences, outcomes, bins),
        "brier": float(np.mean((confidences - outcomes) ** 2)),
        "top1_ece": _ece(top_confidence, top_correct, bins),
        "top1_accuracy": float(top_correct.mean()),
    }


def _ece(confidences: np.ndarray, outcomes: np.ndarray, bins: int) -> float:
    edges = np.minimum((confidences * bins).astype(int), bins - 1)
    error = 0.0
    for bin_index in np.unique(edges):
        members = edges == bin_index
        error += members.mean() * abs(outcomes[members].mean() - confidences[members].mean())
    return float(error)


def close_upward(
    direct: np.ndarray,
    direct_labels: Sequence[str],
    nodes: Sequence[str],
    parents: Mapping[str, str | None],
) -> np.ndarray:
    """Map direct-label indicators into node space, adding every ancestor."""
    column = {node: index for index, node in enumerate(nodes)}
    closed = np.zeros((len(direct), len(nodes)), dtype=np.uint8)
    for source, label in enumerate(direct_labels):
        rows = direct[:, source].astype(bool)
        node: str | None = label
        while node is not None:
            closed[rows, column[node]] = 1
            node = parents[node]
    return closed


def hierarchy_violation_rate(
    closed: np.ndarray, nodes: Sequence[str], parents: Mapping[str, str | None]
) -> float:
    """Fraction of documents with a predicted node whose parent is not predicted."""
    column = {node: index for index, node in enumerate(nodes)}
    violated = np.zeros(len(closed), dtype=bool)
    for child, node in enumerate(nodes):
        parent = parents[node]
        if parent is not None:
            violated |= closed[:, child].astype(bool) & ~closed[:, column[parent]].astype(bool)
    return float(violated.mean())


def ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict[str, float]:
    """AUROC and FPR at 95% TPR, with OOD inputs as the positive class."""
    if not len(id_scores) or not len(ood_scores):
        raise ValueError("Both in-distribution and OOD scores are required.")
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    # Keep every threshold: collinear points dropped by default can hold the minimum FPR.
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "fpr_at_95_tpr": float(fpr[tpr >= 0.95].min()),
    }
