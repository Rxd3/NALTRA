"""Minimal orchestration around the shared model interface."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from naltra.models.base import BaseNALTRAModel
from naltra.pipeline.preprocessing import normalize_text
from naltra.schemas.prediction import PredictionResult


@dataclass(slots=True)
class PredictionPipeline:
    """Prepare text and delegate prediction to the selected model."""

    model: BaseNALTRAModel
    preprocessor: Callable[[str], str] = normalize_text

    def predict(self, text: str) -> PredictionResult:
        prepared = self.preprocessor(text)
        if not prepared:
            raise ValueError("Prediction text cannot be empty.")
        return self.model.predict(prepared)

    def predict_batch(self, texts: Iterable[str]) -> list[PredictionResult]:
        prepared = [self.preprocessor(text) for text in texts]
        if any(not text for text in prepared):
            raise ValueError("Prediction texts cannot contain empty values.")
        return self.model.predict_batch(prepared)
