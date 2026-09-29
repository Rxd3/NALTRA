"""Training entry point for the future Transformer implementation."""

from typing import Any

from naltra.models.transformer.model import TransformerModel


def train_model(train_data: Any, validation_data: Any | None = None) -> TransformerModel:
    model = TransformerModel()
    model.train(train_data, validation_data)
    return model
