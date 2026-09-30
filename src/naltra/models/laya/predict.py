"""Prediction adapter for Laya."""

from naltra.models.laya.model import LayaModel
from naltra.schemas.prediction import PredictionResult


def predict_text(model: LayaModel, text: str) -> PredictionResult:
    return model.predict(text)
