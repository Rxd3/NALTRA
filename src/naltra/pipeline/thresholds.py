"""Multi-label threshold helpers."""

from collections.abc import Mapping

from naltra.schemas.prediction import LabelScore


def apply_thresholds(
    scores: Mapping[str, float],
    default_threshold: float = 0.5,
    per_label: Mapping[str, float] | None = None,
) -> list[LabelScore]:
    """Select labels by global or per-label thresholds, highest score first."""
    if not 0.0 <= default_threshold <= 1.0:
        raise ValueError("default_threshold must be between 0.0 and 1.0")
    thresholds = per_label or {}
    selected = [
        LabelScore(label=label, score=score)
        for label, score in scores.items()
        if score >= thresholds.get(label, default_threshold)
    ]
    return sorted(selected, key=lambda item: item.score, reverse=True)
