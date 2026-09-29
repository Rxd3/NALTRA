"""Small OOD decision helpers; full benchmark metrics will be added later."""

from naltra.schemas.prediction import OODResult


def threshold_ood(score: float, threshold: float = 0.5) -> OODResult:
    """Convert a normalized OOD score into a decision."""
    if not 0.0 <= score <= 1.0 or not 0.0 <= threshold <= 1.0:
        raise ValueError("score and threshold must be between 0.0 and 1.0")
    return OODResult(is_ood=score >= threshold, score=score)
