"""Laya placeholder implementing the shared model contract."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from naltra.models.base import BaseNALTRAModel
from naltra.models.laya.client import LayaClient
from naltra.schemas.prediction import PredictionResult


class LayaModel(BaseNALTRAModel):
    """Future adapter from Laya responses to PredictionResult."""

    def __init__(self, client: LayaClient | None = None) -> None:
        self.client = client

    def train(self, train_data: Any, validation_data: Any | None = None) -> None:
        raise NotImplementedError("Laya training or configuration is not implemented.")

    def predict(self, text: str) -> PredictionResult:
        raise NotImplementedError("Laya prediction has not been implemented.")

    def predict_batch(self, texts: Iterable[str]) -> list[PredictionResult]:
        raise NotImplementedError("Laya batch prediction has not been implemented.")

    def save(self, path: str | Path) -> None:
        raise NotImplementedError("Laya configuration persistence has not been implemented.")

    def load(self, path: str | Path) -> None:
        raise NotImplementedError("Laya configuration loading has not been implemented.")
