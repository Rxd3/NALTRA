"""Out-of-distribution detection boundary."""

from typing import Protocol

from naltra.schemas.prediction import OODResult


class OODDetector(Protocol):
    """Protocol for model-specific or shared OOD detectors."""

    def detect(self, text: str) -> OODResult:
        """Return an OOD decision and normalized score."""
        ...


class DisabledOODDetector:
    """Placeholder that makes the absence of an OOD detector explicit."""

    def detect(self, text: str) -> OODResult:
        del text
        return OODResult(is_ood=False, score=0.0)
