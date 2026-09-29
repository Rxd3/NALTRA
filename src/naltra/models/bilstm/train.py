"""Training entry point for the future BiLSTM implementation."""

from typing import Any

from naltra.models.bilstm.model import BiLSTMModel


def train_model(train_data: Any, validation_data: Any | None = None) -> BiLSTMModel:
    model = BiLSTMModel()
    model.train(train_data, validation_data)
    return model
