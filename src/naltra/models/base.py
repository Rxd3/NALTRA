"""Common interface implemented by every NALTRA model family."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from naltra.schemas.prediction import PredictionResult


class BaseNALTRAModel(ABC):
    """Stable model boundary used by pipelines, evaluation, and the dashboard."""

    @abstractmethod
    def train(self, train_data: Any, validation_data: Any | None = None) -> None:
        """Fit model state from prepared training data."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, text: str) -> PredictionResult:
        """Return a standardized prediction for one text."""
        raise NotImplementedError

    @abstractmethod
    def predict_batch(self, texts: Iterable[str]) -> list[PredictionResult]:
        """Return standardized predictions in the same order as the inputs."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist fitted model state and required metadata."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load fitted model state into this instance."""
        raise NotImplementedError
