"""Prediction adapter for Naive Bayes."""

from naltra.models.naive_bayes.model import NaiveBayesModel
from naltra.schemas.prediction import PredictionResult


def predict_text(model: NaiveBayesModel, text: str) -> PredictionResult:
    return model.predict(text)
