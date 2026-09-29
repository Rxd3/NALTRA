"""Prediction adapter for Jev."""

from naltra.models.jev.model import JevModel
from naltra.schemas.prediction import PredictionResult


def predict_text(model: JevModel, text: str) -> PredictionResult:
    return model.predict(text)
