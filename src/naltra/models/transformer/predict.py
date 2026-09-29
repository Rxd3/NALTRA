"""Prediction adapter for the multilingual Transformer."""

from naltra.models.transformer.model import TransformerModel
from naltra.schemas.prediction import PredictionResult


def predict_text(model: TransformerModel, text: str) -> PredictionResult:
    return model.predict(text)
