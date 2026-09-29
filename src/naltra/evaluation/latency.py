"""Local latency measurement utilities."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import TypeVar

T = TypeVar("T")


def measure_latency_ms(operation: Callable[[], T]) -> tuple[T, float]:
    """Run an operation once and return its result and elapsed milliseconds."""
    start = perf_counter()
    result = operation()
    return result, (perf_counter() - start) * 1000.0
