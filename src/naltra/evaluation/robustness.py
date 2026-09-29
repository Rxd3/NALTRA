"""Prediction stability helpers."""


def jaccard_stability(original: set[str], perturbed: set[str]) -> float:
    """Measure label-set stability; two empty sets are perfectly stable."""
    union = original | perturbed
    return len(original & perturbed) / len(union) if union else 1.0
