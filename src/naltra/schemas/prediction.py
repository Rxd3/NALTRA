"""Typed, model-independent prediction structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any


def _validate_probability(value: float, field_name: str) -> None:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.")


@dataclass(frozen=True, slots=True)
class LanguageInfo:
    """Detected language information for one input."""

    primary: str
    is_code_switched: bool = False

    def __post_init__(self) -> None:
        if not self.primary.strip():
            raise ValueError("Language primary code cannot be empty.")


@dataclass(frozen=True, slots=True)
class LabelScore:
    """One canonical label and its confidence score."""

    label: str
    score: float

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("Label cannot be empty.")
        _validate_probability(self.score, "Label score")


@dataclass(frozen=True, slots=True)
class OODResult:
    """Out-of-distribution decision and normalized OOD score."""

    is_ood: bool
    score: float

    def __post_init__(self) -> None:
        _validate_probability(self.score, "OOD score")


@dataclass(frozen=True, slots=True)
class Explanation:
    """Model explanation payload that can grow without changing core fields."""

    important_tokens: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Standard output returned by every NALTRA model."""

    text: str
    model: str
    language: LanguageInfo
    labels: list[LabelScore] = field(default_factory=list)
    hierarchy_paths: list[list[str]] = field(default_factory=list)
    ood: OODResult = field(default_factory=lambda: OODResult(is_ood=False, score=0.0))
    latency_ms: float = 0.0
    explanation: Explanation | None = None
    label_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("Model name cannot be empty.")
        if not isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("Latency must be finite and cannot be negative.")
        if len({item.label for item in self.labels}) != len(self.labels):
            raise ValueError("Prediction labels must be unique.")
        for label, score in self.label_scores.items():
            LabelScore(label=label, score=score)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)
