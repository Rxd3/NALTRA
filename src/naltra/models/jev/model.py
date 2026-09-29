"""Jev placeholder implementing the shared model contract."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from naltra.models.base import BaseNALTRAModel
from naltra.models.jev.client import JevClient
from naltra.schemas.prediction import PredictionResult


class JevModel(BaseNALTRAModel):
    """Future adapter from Jev responses to PredictionResult."""

    def __init__(self, client: JevClient | None = None) -> None:
        self.client = client

    def train(self, train_data: Any, validation_data: Any | None = None) -> None:
        raise NotImplementedError("Jev training or configuration is not implemented.")

    def predict(self, text: str) -> PredictionResult:
        raise NotImplementedError("Jev prediction has not been implemented.")

    def predict_batch(self, texts: Iterable[str]) -> list[PredictionResult]:
        raise NotImplementedError("Jev batch prediction has not been implemented.")

    def save(self, path: str | Path) -> None:
        raise NotImplementedError("Jev configuration persistence has not been implemented.")

    def load(self, path: str | Path) -> None:
        raise NotImplementedError("Jev configuration loading has not been implemented.")
