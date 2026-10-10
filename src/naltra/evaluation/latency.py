"""Local latency measurement utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import cycle, islice
from time import perf_counter
from typing import TypeVar

import numpy as np

T = TypeVar("T")


def latency_profile(
    operation: Callable[[T], object], inputs: Sequence[T], warmup: int = 10
) -> dict[str, float]:
    """Per-input latency percentiles after ``warmup`` untimed calls, cycling inputs as needed."""
    if not inputs or warmup < 0:
        raise ValueError("Latency profiling needs inputs and a nonnegative warmup count.")
    for item in islice(cycle(inputs), warmup):
        operation(item)
    timings = []
    for item in inputs:
        start = perf_counter()
        operation(item)
        timings.append((perf_counter() - start) * 1000.0)
    p50, p95, p99 = np.percentile(timings, [50, 95, 99])
    return {
        "samples": len(timings),
        "warmup": warmup,
        "mean_ms": float(np.mean(timings)),
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
    }
