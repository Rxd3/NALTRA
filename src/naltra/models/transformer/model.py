"""Hugging Face multi-label classifier with local artifact reload."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from naltra.models.neural import NeuralModel


class TransformerModel(NeuralModel):
    model_name = "transformer"

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        taxonomy_path: str | Path | None = None,
        tokenizer: Any = None,
        network: nn.Module | None = None,
    ) -> None:
        super().__init__(config, taxonomy_path=taxonomy_path)
        self._validate_architecture()
        if (tokenizer is None) != (network is None):
            raise ValueError("Provide both tokenizer and network, or neither.")
        self.tokenizer = tokenizer
        self.network = network

    def _validate_architecture(self) -> None:
        if not self.config["architecture"]["pretrained_name"]:
            raise ValueError("architecture.pretrained_name cannot be empty.")
        if type(self.config["architecture"]["local_files_only"]) is not bool:
            raise ValueError("architecture.local_files_only must be a boolean.")

    def _initialize(self, texts: Sequence[str]) -> None:
        del texts
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        architecture = self.config["architecture"]
        if self.network is None:
            source = architecture["pretrained_name"]
            options = {"local_files_only": architecture.get("local_files_only", False)}
            if architecture.get("revision"):
                options["revision"] = architecture["revision"]
            self.tokenizer = AutoTokenizer.from_pretrained(source, **options)
            self.network = AutoModelForSequenceClassification.from_pretrained(
                source,
                num_labels=len(self.labels),
                problem_type="multi_label_classification",
                id2label=dict(enumerate(self.labels)),
                label2id={label: i for i, label in enumerate(self.labels)},
                **options,
            )
        elif self.network.config.num_labels != len(self.labels):
            raise ValueError("Injected network head does not match training label space.")
        self.network.config.problem_type = "multi_label_classification"
        self.network.config.id2label = dict(enumerate(self.labels))
        self.network.config.label2id = {label: i for i, label in enumerate(self.labels)}
        self.metadata["pretrained_revision"] = getattr(self.network.config, "_commit_hash", None)
        if self.config["training"].get("gradient_checkpointing", True):
            self.network.gradient_checkpointing_enable()

    def _encode(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("Transformer tokenizer is unavailable.")
        return dict(
            self.tokenizer(
                list(texts),
                truncation=True,
                padding=True,
                max_length=self.config["architecture"]["max_length"],
                return_tensors="pt",
            )
        )

    def _logits(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        assert self.network is not None
        return self.network(**inputs).logits

    def _save_network(self, target: Path) -> None:
        assert self.network is not None
        self.network.save_pretrained(target, safe_serialization=True)
        self.tokenizer.save_pretrained(target)

    def _load_network(self, target: Path) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(target, local_files_only=True)
        self.network = AutoModelForSequenceClassification.from_pretrained(
            target, local_files_only=True
        )
        if (
            self.network.config.num_labels != len(self.labels)
            or [self.network.config.id2label.get(index) for index in range(len(self.labels))]
            != self.labels
        ):
            raise ValueError("Transformer artifact head/labels do not match metadata.")
