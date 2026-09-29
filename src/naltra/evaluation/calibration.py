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
