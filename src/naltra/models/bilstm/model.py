"""BiLSTM placeholder implementing the shared model contract."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from naltra.models.base import BaseNALTRAModel
from naltra.schemas.prediction import PredictionResult


class BiLSTMModel(BaseNALTRAModel):
    """Future multilingual BiLSTM implementation."""

    def train(self, train_data: Any, validation_data: Any | None = None) -> None:
        raise NotImplementedError("BiLSTM training has not been implemented.")

    def predict(self, text: str) -> PredictionResult:
        raise NotImplementedError("BiLSTM prediction has not been implemented.")

    def predict_batch(self, texts: Iterable[str]) -> list[PredictionResult]:
        raise NotImplementedError("BiLSTM batch prediction has not been implemented.")

    def save(self, path: str | Path) -> None:
        raise NotImplementedError("BiLSTM persistence has not been implemented.")

    def load(self, path: str | Path) -> None:
        raise NotImplementedError("BiLSTM loading has not been implemented.")
