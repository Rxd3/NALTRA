"""Prediction adapter for SVM."""

from naltra.models.svm.model import SVMModel
from naltra.schemas.prediction import PredictionResult


def predict_text(model: SVMModel, text: str) -> PredictionResult:
    return model.predict(text)
