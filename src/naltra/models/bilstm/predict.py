"""Prediction adapter for BiLSTM."""

from naltra.models.bilstm.model import BiLSTMModel
from naltra.schemas.prediction import PredictionResult


def predict_text(model: BiLSTMModel, text: str) -> PredictionResult:
    return model.predict(text)
