"""Out-of-distribution detection boundary."""

from collections.abc import Mapping
from math import isfinite
from typing import Protocol

from naltra.schemas.prediction import OODResult


class OODDetector(Protocol):
    """Protocol for model-specific or shared OOD detectors."""

    def detect(self, text: str) -> OODResult:
        """Return an OOD decision and normalized score."""
        ...


class MaxProbabilityOODDetector:
    """Confidence baseline; the decision threshold comes from ID validation."""

    def __init__(self, threshold: float) -> None:
        if not isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("OOD threshold must be finite and between 0 and 1.")
        self.threshold = threshold

    def detect_scores(self, scores: Mapping[str, float]) -> OODResult:
        if not scores:
            raise ValueError("Confidence OOD detection requires complete label scores.")
        if any(not isfinite(score) or not 0 <= score <= 1 for score in scores.values()):
            raise ValueError("Invalid OOD input probabilities.")
        score = 1.0 - max(scores.values())
        return OODResult(is_ood=score > self.threshold, score=score)
