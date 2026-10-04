"""Training entry point for the future BiLSTM implementation."""

from typing import Any

from naltra.models.bilstm.model import BiLSTMModel


def train_model(train_data: Any, validation_data: Any | None = None, **kwargs: Any) -> BiLSTMModel:
    model = BiLSTMModel(**kwargs)
    model.train(train_data, validation_data)
    return model
