"""Unicode vocabulary and packed BiLSTM multi-label classifier."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from naltra.models.neural import NeuralModel


def tokenize(text: str) -> list[str]:
    """Preserve casing and Turkish characters."""
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


class BiLSTMNetwork(nn.Module):
    def __init__(self, vocab_size: int, label_count: int, architecture: Mapping[str, Any]) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, architecture["embedding_dim"], padding_idx=0)
        self.lstm = nn.LSTM(
            architecture["embedding_dim"],
            architecture["hidden_dim"],
            num_layers=architecture["layers"],
            bidirectional=True,
            batch_first=True,
            dropout=architecture["dropout"] if architecture["layers"] > 1 else 0.0,
        )
        self.dropout = nn.Dropout(architecture["dropout"])
        self.head = nn.Linear(2 * architecture["hidden_dim"], label_count)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            self.embedding(input_ids), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return self.head(self.dropout(torch.cat((hidden[-2], hidden[-1]), dim=1)))


class BiLSTMModel(NeuralModel):
    model_name = "bilstm"

    def __init__(
        self, config: Mapping[str, Any] | None = None, *, taxonomy_path: str | Path | None = None
    ) -> None:
        super().__init__(config, taxonomy_path=taxonomy_path)
        self._validate_architecture()
        self.vocabulary: dict[str, int] = {}

    def _validate_architecture(self) -> None:
        architecture = self.config["architecture"]
        for key in ("embedding_dim", "hidden_dim", "layers", "max_vocab"):
            if type(architecture[key]) is not int or architecture[key] < 1:
                raise ValueError(f"architecture.{key} must be a positive integer.")
        if architecture["max_vocab"] < 3:
            raise ValueError("max_vocab must leave room for padding, unknown and text tokens.")
        if not 0 <= architecture["dropout"] < 1:
            raise ValueError("dropout must be in [0, 1).")

    def _initialize(self, texts: Sequence[str]) -> None:
        counts = Counter(token for text in texts for token in tokenize(text))
        tokens = sorted(counts, key=lambda token: (-counts[token], token))
        self.vocabulary = {"<PAD>": 0, "<UNK>": 1}
        for token in tokens[: self.config["architecture"]["max_vocab"] - 2]:
            self.vocabulary[token] = len(self.vocabulary)
        self.network = BiLSTMNetwork(
            len(self.vocabulary), len(self.labels), self.config["architecture"]
        )

    def _encode(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        limit = self.config["architecture"]["max_length"]
        ids = [
            [self.vocabulary.get(token, 1) for token in tokenize(text)[:limit]] or [1]
            for text in texts
        ]
        width = max(map(len, ids), default=1)
        return {
            "input_ids": torch.tensor(
                [row + [0] * (width - len(row)) for row in ids], dtype=torch.long
            ),
            "lengths": torch.tensor(list(map(len, ids)), dtype=torch.long),
        }

    def _logits(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        assert self.network is not None
        return self.network(**inputs)

    def _save_network(self, target: Path) -> None:
        assert self.network is not None
        torch.save(self.network.state_dict(), target / "weights.pt")
        (target / "vocabulary.json").write_text(
            json.dumps(self.vocabulary, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def _load_network(self, target: Path) -> None:
        self.vocabulary = json.loads((target / "vocabulary.json").read_text(encoding="utf-8"))
        if (
            self.vocabulary.get("<PAD>") != 0
            or self.vocabulary.get("<UNK>") != 1
            or set(self.vocabulary.values()) != set(range(len(self.vocabulary)))
        ):
            raise ValueError("Invalid artifact vocabulary.")
        self.network = BiLSTMNetwork(
            len(self.vocabulary), len(self.labels), self.config["architecture"]
        )
        self.network.load_state_dict(
            torch.load(target / "weights.pt", map_location="cpu", weights_only=True)
        )
