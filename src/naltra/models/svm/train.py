"""Training entry point for the future SVM implementation."""

from typing import Any

from naltra.models.svm.model import SVMModel


def train_model(train_data: Any, validation_data: Any | None = None) -> SVMModel:
    model = SVMModel()
    model.train(train_data, validation_data)
    return model
