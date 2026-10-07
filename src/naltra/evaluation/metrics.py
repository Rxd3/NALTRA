"""Dependency-free metric helpers suitable for unit tests and baselines."""

from __future__ import annotations

from collections.abc import Iterable

from naltra.schemas.prediction import PredictionResult


def safe_divide(numerator: float, denominator: float) -> float:
    """Divide while defining an empty denominator as zero."""
    return numerator / denominator if denominator else 0.0


def precision_recall_f1(
    true_labels: set[str], predicted_labels: set[str]
) -> tuple[float, float, float]:
    """Calculate example-level set precision, recall, and F1."""
    true_positives = len(true_labels & predicted_labels)
    precision = safe_divide(true_positives, len(predicted_labels))
    recall = safe_divide(true_positives, len(true_labels))
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1


def multilabel_metrics(
    y_true: Iterable[set[str]],
    y_pred: Iterable[set[str]],
    all_labels: Iterable[str] | None = None,
) -> dict[str, float]:
    """Calculate micro and macro precision/recall/F1 for multilabel sets."""
    truth = list(y_true)
    predictions = list(y_pred)
    if len(truth) != len(predictions):
        raise ValueError("y_true and y_pred must have the same number of examples.")

    observed = set().union(*truth, *predictions) if truth or predictions else set()
    label_universe = set(all_labels) if all_labels is not None else observed
    if observed - label_universe:
        raise ValueError("Labels fall outside the evaluation label universe.")
    micro_tp = sum(
        len(actual & predicted) for actual, predicted in zip(truth, predictions, strict=False)
    )
    micro_fp = sum(
        len(predicted - actual) for actual, predicted in zip(truth, predictions, strict=False)
    )
    micro_fn = sum(
        len(actual - predicted) for actual, predicted in zip(truth, predictions, strict=False)
    )
    micro_precision = safe_divide(micro_tp, micro_tp + micro_fp)
    micro_recall = safe_divide(micro_tp, micro_tp + micro_fn)
    micro_f1 = safe_divide(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    per_label: list[tuple[float, float, float]] = []
    for label in sorted(label_universe):
        tp = sum(
            label in actual and label in predicted
            for actual, predicted in zip(truth, predictions, strict=False)
        )
        fp = sum(
            label not in actual and label in predicted
            for actual, predicted in zip(truth, predictions, strict=False)
        )
        fn = sum(
            label in actual and label not in predicted
            for actual, predicted in zip(truth, predictions, strict=False)
        )
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        per_label.append((precision, recall, f1))

    macro_precision = safe_divide(sum(item[0] for item in per_label), len(per_label))
    macro_recall = safe_divide(sum(item[1] for item in per_label), len(per_label))
    macro_f1 = safe_divide(sum(item[2] for item in per_label), len(per_label))
    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def compute_classification_metrics(
    predictions: list[PredictionResult],
    ground_truth: list[set[str]],
    all_labels: list[str] | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate multi-label classification metrics for prediction results."""
    y_pred_sets = []
    for pred in predictions:
        pred_set = {item.label for item in pred.labels if item.score >= threshold}
        y_pred_sets.append(pred_set)

    return multilabel_metrics(ground_truth, y_pred_sets, all_labels)
