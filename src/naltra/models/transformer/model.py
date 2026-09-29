"""Transformer placeholder implementing the shared model contract."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from naltra.models.base import BaseNALTRAModel
from naltra.schemas.prediction import PredictionResult


class TransformerModel(BaseNALTRAModel):
    """Future multilingual Transformer implementation."""

    def train(self, train_data: Any, validation_data: Any | None = None) -> None:
        raise NotImplementedError("Transformer training has not been implemented.")

    def predict(self, text: str) -> PredictionResult:
        raise NotImplementedError("Transformer prediction has not been implemented.")

    def predict_batch(self, texts: Iterable[str]) -> list[PredictionResult]:
        raise NotImplementedError("Transformer batch prediction has not been implemented.")

    def save(self, path: str | Path) -> None:
        raise NotImplementedError("Transformer persistence has not been implemented.")

    def load(self, path: str | Path) -> None:
        raise NotImplementedError("Transformer loading has not been implemented.")
