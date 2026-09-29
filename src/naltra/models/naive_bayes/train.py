"""Training entry point for the future Naive Bayes implementation."""

from typing import Any

from naltra.models.naive_bayes.model import NaiveBayesModel


def train_model(train_data: Any, validation_data: Any | None = None) -> NaiveBayesModel:
    model = NaiveBayesModel()
    model.train(train_data, validation_data)
    return model
