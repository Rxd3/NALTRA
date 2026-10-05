"""Confidence calibration helpers."""

from collections.abc import Sequence


def brier_score(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    """Calculate mean squared probability error for binary outcomes."""
    if len(probabilities) != len(outcomes) or not probabilities:
        raise ValueError("Probabilities and outcomes must be non-empty and equally sized.")
    if any(not 0.0 <= probability <= 1.0 for probability in probabilities):
        raise ValueError("Probabilities must be between 0.0 and 1.0.")
    if any(outcome not in (0, 1) for outcome in outcomes):
        raise ValueError("Outcomes must be binary.")
    return sum(
        (probability - outcome) ** 2
        for probability, outcome in zip(probabilities, outcomes, strict=False)
    ) / len(outcomes)


def expected_calibration_error(
    probabilities: Sequence[float], outcomes: Sequence[int], bins: int = 10
) -> float:
    """Calculate equally spaced expected calibration error."""
    if len(probabilities) != len(outcomes) or not probabilities:
        raise ValueError("Probabilities and outcomes must be non-empty and equally sized.")
    if bins <= 0:
        raise ValueError("bins must be positive.")
    if any(not 0.0 <= probability <= 1.0 for probability in probabilities):
        raise ValueError("Probabilities must be between 0.0 and 1.0.")

    total = len(probabilities)
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        members = [
            index
            for index, probability in enumerate(probabilities)
            if lower <= probability < upper or (bin_index == bins - 1 and probability == 1.0)
        ]
        if not members:
            continue
        confidence = sum(probabilities[index] for index in members) / len(members)
        accuracy = sum(outcomes[index] for index in members) / len(members)
        error += len(members) / total * abs(accuracy - confidence)
    return error

import numpy as np
from typing import List, Dict
from naltra.schemas import PredictionResult

def compute_calibration_ece(
    predictions: List[PredictionResult],
    ground_truth: List[Dict[str, int]],
    n_bins: int = 10
) -> Dict[str, float]:
    """Computes Expected Calibration Error (ECE)."""
    confidences = []
    accuracies = []

    for pred, gt in zip(predictions, ground_truth):
        for item in pred.labels:
            confidences.append(item.score)
            is_correct = 1.0 if gt.get(item.label, 0) == 1 else 0.0
            accuracies.append(is_correct)

    if not confidences:
        return {"ece": 0.0}

    conf_arr = np.array(confidences)
    acc_arr = np.array(accuracies)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (conf_arr > bin_lower) & (conf_arr <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(acc_arr[in_bin])
            avg_confidence_in_bin = np.mean(conf_arr[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return {"ece": float(ece)}
    