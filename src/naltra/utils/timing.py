"""Reusable timing context manager."""

from dataclasses import dataclass, field
from time import perf_counter


@dataclass(slots=True)
class Timer:
    """Record elapsed milliseconds around a context block."""

    elapsed_ms: float = field(default=0.0, init=False)
    _started_at: float = field(default=0.0, init=False, repr=False)

    def __enter__(self) -> "Timer":
        self._started_at = perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.elapsed_ms = (perf_counter() - self._started_at) * 1000.0
