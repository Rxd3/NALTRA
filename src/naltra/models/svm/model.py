"""SVM placeholder implementing the shared model contract."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from naltra.models.base import BaseNALTRAModel
from naltra.schemas.prediction import PredictionResult


class SVMModel(BaseNALTRAModel):
    """Future TF-IDF + SVM multi-label implementation."""

    def train(self, train_data: Any, validation_data: Any | None = None) -> None:
        raise NotImplementedError("SVM training has not been implemented.")

    def predict(self, text: str) -> PredictionResult:
        raise NotImplementedError("SVM prediction has not been implemented.")

    def predict_batch(self, texts: Iterable[str]) -> list[PredictionResult]:
        raise NotImplementedError("SVM batch prediction has not been implemented.")

    def save(self, path: str | Path) -> None:
        raise NotImplementedError("SVM persistence has not been implemented.")

    def load(self, path: str | Path) -> None:
        raise NotImplementedError("SVM loading has not been implemented.")
