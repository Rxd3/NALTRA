"""Vectorized paper metrics over (documents x labels) matrices."""

from __future__ import annotations

import numpy as np
import pytest

from naltra.evaluation.matrix import (
    binarize,
    calibration_metrics,
    close_upward,
    flat_metrics,
    hierarchy_violation_rate,
    label_distribution,
    ood_metrics,
    ranking_metrics,
)

LABELS = ["a", "b", "c"]


def test_binarize_ignores_label_order_and_rejects_unknown_labels() -> None:
    matrix = binarize([["b"], ["c", "a"]], LABELS)
    assert matrix.tolist() == [[0, 1, 0], [1, 0, 1]]
    with pytest.raises(ValueError, match="unknown"):
        binarize([["z"]], LABELS)


def test_macro_average_covers_the_full_label_universe() -> None:
    truth = binarize([["a"], ["a"]], LABELS)
    predicted = binarize([["a"], ["a"]], LABELS)
    metrics = flat_metrics(truth, predicted)
    assert metrics["micro_f1"] == pytest.approx(1.0)
    # Labels b and c never occur: they count as zero, not as missing.
    assert metrics["macro_f1"] == pytest.approx(1 / 3)
    assert metrics["jaccard"] == pytest.approx(1.0)
    assert metrics["mean_predicted_labels"] == pytest.approx(1.0)


def test_flat_metrics_count_partial_matches() -> None:
    truth = binarize([["a", "b"], ["c"]], LABELS)
    predicted = binarize([["a"], ["b", "c"]], LABELS)
    metrics = flat_metrics(truth, predicted)
    assert metrics["micro_precision"] == pytest.approx(2 / 3)
    assert metrics["micro_recall"] == pytest.approx(2 / 3)
    assert metrics["jaccard"] == pytest.approx((1 / 2 + 1 / 2) / 2)
    assert metrics["document_f1"] == pytest.approx((2 / 3 + 2 / 3) / 2)


def test_ranking_metrics() -> None:
    truth = binarize([["a", "b"], ["c"]], LABELS)
    scores = np.array([[0.9, 0.1, 0.8], [0.2, 0.3, 0.7]])
    metrics = ranking_metrics(truth, scores, ks=(1, 2))
    assert metrics["p_at_1"] == pytest.approx(1.0)
    assert metrics["p_at_2"] == pytest.approx((1 / 2 + 1 / 2) / 2)
    # R-precision: top-2 of doc 1 holds one of two true labels; top-1 of doc 2 is right.
    assert metrics["r_precision"] == pytest.approx((1 / 2 + 1) / 2)
    assert 0 < metrics["ndcg_at_5"] <= 1


def test_calibration_uses_every_cell_not_only_selected_labels() -> None:
    truth = binarize([["a"], ["b"]], LABELS)
    perfect = truth.astype(float)
    assert calibration_metrics(truth, perfect)["ece"] == pytest.approx(0.0)
    assert calibration_metrics(truth, perfect)["brier"] == pytest.approx(0.0)
    overconfident = np.full((2, 3), 0.9)
    metrics = calibration_metrics(truth, overconfident, bins=10)
    # Every cell sits in the top bin: confidence 0.9, positive rate 2/6.
    assert metrics["ece"] == pytest.approx(0.9 - 2 / 6)
    assert metrics["top1_accuracy"] == pytest.approx(1 / 2)


def test_close_upward_adds_ancestors_in_node_space() -> None:
    parents = {"root": None, "mid": "root", "leaf": "mid", "other": "root"}
    nodes = ["root", "mid", "leaf", "other"]
    direct = binarize([["leaf"], ["other"]], ["leaf", "other"])
    closed = close_upward(direct, ["leaf", "other"], nodes, parents)
    assert closed.tolist() == [[1, 1, 1, 0], [1, 0, 0, 1]]
    assert hierarchy_violation_rate(closed, nodes, parents) == 0.0
    broken = np.array([[0, 1, 1, 0], [1, 0, 0, 1]])
    assert hierarchy_violation_rate(broken, nodes, parents) == pytest.approx(0.5)


def test_ood_metrics_treat_ood_as_the_positive_class() -> None:
    metrics = ood_metrics(np.array([0.1, 0.2, 0.3]), np.array([0.8, 0.9, 0.25]))
    assert metrics["auroc"] == pytest.approx(8 / 9)
    # Catching the 0.25 OOD input also flags the 0.3 ID input.
    assert metrics["fpr_at_95_tpr"] == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        ood_metrics(np.array([]), np.array([0.5]))


def test_fpr_at_95_tpr_uses_the_best_roc_operating_point() -> None:
    # Threshold 0.05 keeps 19 of 20 OOD scores (TPR 0.95) and no ID score.
    assert ood_metrics(np.array([0.025]), np.arange(20) / 20)["fpr_at_95_tpr"] == 0.0
    # Tied scores cannot be split: reaching TPR 0.95 accepts every ID score at 0.5 too.
    tied = ood_metrics(np.array([0.1, 0.5, 0.5, 0.9]), np.array([0.5] * 19 + [1.0]))
    assert tied["fpr_at_95_tpr"] == pytest.approx(0.75)
    # (0, 0.9), (0.05, 0.95), (0.1, 1.0) are collinear; the middle point is the answer.
    collinear = ood_metrics(np.array([0.8, 0.7] + [0.0] * 18), np.array([1.0] * 18 + [0.8, 0.7]))
    assert collinear["fpr_at_95_tpr"] == pytest.approx(0.05)


def test_label_distribution_reports_the_imbalance_that_makes_accuracy_misleading() -> None:
    # Column j is positive in its first counts[j] of 200 records.
    counts = np.array([100, 50, 20, 10, 4, 2, 1, 1, 0, 0])
    truth = (np.arange(200)[:, None] < counts[None, :]).astype(np.uint8)
    assert label_distribution(truth) == pytest.approx(
        {
            "n_records": 200,
            "n_labels": 10,
            "mean_labels_per_record": 188 / 200,
            "median_labels_per_record": 0.5,
            "min_prevalence": 0.0,
            "median_prevalence": 0.015,
            "max_prevalence": 0.5,
            "rare_labels": 4,
            "rare_label_share": 0.4,
            "top_decile_positive_share": 100 / 188,
            "prevalence_ratio": 100.0,
            "zero_positive_labels": 2,
            "all_negative_accuracy": 1 - 188 / 2000,
        }
    )
    empty = label_distribution(np.zeros((3, 2), dtype=np.uint8))
    assert empty["prevalence_ratio"] is None and empty["top_decile_positive_share"] == 0.0
    with pytest.raises(ValueError):
        label_distribution(np.zeros((0, 2), dtype=np.uint8))
