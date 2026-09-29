"""Shared records for future reproducible benchmark outputs."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    """Minimal serializable summary for one model and evaluation slice."""

    model: str
    split: str
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
