"""Multi-label threshold helpers."""

from collections.abc import Mapping
from numbers import Real
from typing import Any

from naltra.schemas.prediction import LabelScore


def is_probability(value: Any) -> bool:
    """A real number in [0, 1]; bools, numeric text, None, NaN and infinities are not."""
    return isinstance(value, Real) and not isinstance(value, bool) and 0 <= value <= 1


def apply_thresholds(
    scores: Mapping[str, float],
    default_threshold: float = 0.5,
    per_label: Mapping[str, float] | None = None,
) -> list[LabelScore]:
    """Select labels by global or per-label thresholds, highest score first."""
    if not is_probability(default_threshold):
        raise ValueError("default_threshold must be a number between 0.0 and 1.0")
    thresholds = per_label or {}
    if not isinstance(thresholds, Mapping) or not all(map(is_probability, thresholds.values())):
        raise ValueError("Per-label thresholds must be numbers between 0.0 and 1.0")
    for label, threshold in thresholds.items():
        LabelScore(label=label, score=threshold)
    for label, score in scores.items():
        LabelScore(label=label, score=score)
    selected = [
        LabelScore(label=label, score=score)
        for label, score in scores.items()
        if score >= thresholds.get(label, default_threshold)
    ]
    return sorted(selected, key=lambda item: item.score, reverse=True)
