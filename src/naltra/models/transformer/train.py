"""Training entry point for the future Transformer implementation."""

from typing import Any

from naltra.models.transformer.model import TransformerModel


def train_model(
    train_data: Any, validation_data: Any | None = None, **kwargs: Any
) -> TransformerModel:
    model = TransformerModel(**kwargs)
    model.train(train_data, validation_data)
    return model
