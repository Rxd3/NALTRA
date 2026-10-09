"""Pinned multilingual Laya as an independent binary classifier per CORDIS topic."""

from __future__ import annotations

import copy
import json
import re
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

from naltra.data.manifest import REPO_ROOT, compute_file_sha256, get_taxonomy_checksums
from naltra.models.base import BaseNALTRAModel
from naltra.models.laya.client import CHECKPOINT_FILES, LayaClient
from naltra.models.neural import merge_config, select_device
from naltra.pipeline.preprocessing import normalize_text
from naltra.pipeline.thresholds import apply_thresholds
from naltra.schemas.prediction import LanguageInfo, PredictionResult
from naltra.utils.config import load_yaml


class LayaModel(BaseNALTRAModel):
    model_name = "laya"

    def __init__(
        self, config: Mapping[str, Any] | None = None, *, client: LayaClient | None = None
    ) -> None:
        self.config = merge_config(load_yaml(REPO_ROOT / "configs/models/laya.yaml"), config or {})
        self._validate_config()
        self.device = select_device(self.config["device"])
        self.taxonomy_identity = get_taxonomy_checksums(REPO_ROOT / "taxonomy")
        taxonomy = json.loads((REPO_ROOT / "taxonomy/taxonomy.json").read_text(encoding="utf-8"))
        self.parents = {item["id"]: item["parent"] for item in taxonomy["labels"]}
        direct = sorted(item["id"] for item in taxonomy["labels"] if item["is_direct_supported"])
        self.labels = self.config["supported_labels"] or direct
        if (
            not isinstance(self.labels, list)
            or not self.labels
            or any(not isinstance(label, str) or label not in direct for label in self.labels)
            or len(set(self.labels)) != len(self.labels)
        ):
            raise ValueError(
                "Laya supported_labels must be unique supported direct taxonomy labels."
            )
        self.labels = sorted(self.labels)
        if set(self.config["multilabel"]["per_label"]) - set(self.labels):
            raise ValueError("Laya thresholds contain unsupported labels.")
        by_id = {item["id"]: item for item in taxonomy["labels"]}
        self.questions = {
            label: {
                "type": "choice",
                "instructions": (
                    "Which topic best matches the research described in this project?"
                ),
                "criteria": {
                    "A": by_id[label]["name"],
                    "B": "other unrelated research topics",
                },
            }
            for label in self.labels
        }
        self.metadata: dict[str, Any] = {
            "model_kind": "pretrained_zero_shot",
            "protocol": "independent topic-name versus unrelated-topics choice; score=P(A)",
            "calibration": "uncalibrated_pretrained_probabilities",
        }
        self.client = client or LayaClient({**self.config, "device": str(self.device)})

    def _validate_config(self) -> None:
        allowed = {
            "model",
            "enabled",
            "device",
            "seed",
            "target_field",
            "sdk_version",
            "checkpoint",
            "inference",
            "supported_labels",
            "multilabel",
        }
        if set(self.config) != allowed or self.config["model"] != "laya":
            raise ValueError("Invalid local Laya configuration; API credentials are not used.")
        if type(self.config["enabled"]) is not bool:
            raise ValueError("Laya enabled must be a boolean.")
        if type(self.config["seed"]) is not int or not 0 <= self.config["seed"] < 2**32:
            raise ValueError("Invalid Laya seed.")
        if self.config["target_field"] != "labels_direct" or self.config["sdk_version"] != "0.4.1":
            raise ValueError("Laya requires direct targets and the supported SDK version 0.4.1.")
        checkpoint = self.config["checkpoint"]
        if (
            set(checkpoint) != {"repo_id", "subfolder", "revision"}
            or checkpoint["repo_id"] != "convaiinnovations/laya"
            or checkpoint["subfolder"] != "multilingual"
            or not isinstance(checkpoint["revision"], str)
            or not re.fullmatch(r"[0-9a-f]{40}", checkpoint["revision"])
        ):
            raise ValueError("Laya requires a pinned multilingual checkpoint commit.")
        inference = self.config["inference"]
        if set(inference) != {
            "batch_size",
            "question_batch_size",
            "max_length",
            "head_max_length",
        } or any(type(value) is not int or value < 1 for value in inference.values()):
            raise ValueError("Laya inference sizes must be positive integers.")
        if not 32 <= inference["head_max_length"] < inference["max_length"] <= 8192:
            raise ValueError("Invalid Laya document/question token budget.")
        if not isinstance(self.config["supported_labels"], list):
            raise ValueError("Laya supported_labels must be a list.")
        apply_thresholds(
            {}, self.config["multilabel"]["threshold"], self.config["multilabel"]["per_label"]
        )

    def bind_release(self, release: dict[str, Any]) -> None:
        if self.labels != release["label_universe"]:
            raise ValueError("Laya must cover the exact CORDIS direct-label universe.")
        self.metadata.update(
            {
                "dataset": "cordis_h2020",
                "source_languages": sorted(release["manifests"]),
                "dataset_manifests": copy.deepcopy(release["manifests"]),
                "training_performed": False,
            }
        )

    def train(self, train_data: Any, validation_data: Any | None = None) -> None:
        raise RuntimeError(
            "This Laya integration uses pretrained inference; local training is unsupported."
        )

    def predict(self, text: str) -> PredictionResult:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: Iterable[str]) -> list[PredictionResult]:
        if not self.config["enabled"]:
            raise RuntimeError("Laya is disabled.")
        prepared = [normalize_text(text) for text in texts]
        if any(not text for text in prepared):
            raise ValueError("Prediction texts cannot contain empty values.")
        if not prepared:
            return []
        self.client.initialize()
        started = perf_counter()
        scores = self.client.predict_batch(prepared, self.questions)
        if len(scores) != len(prepared) or any(set(score) != set(self.labels) for score in scores):
            raise ValueError("Laya must return every direct-topic probability for every text.")
        latency = (perf_counter() - started) * 1000 / len(prepared)
        return [
            PredictionResult(
                text=text,
                model="laya",
                language=LanguageInfo("und"),
                label_scores=score,
                labels=apply_thresholds(
                    score,
                    self.config["multilabel"]["threshold"],
                    self.config["multilabel"]["per_label"],
                ),
                latency_ms=latency,
                metadata={
                    "model_kind": self.metadata["model_kind"],
                    "calibration": self.metadata["calibration"],
                    "protocol": self.metadata["protocol"],
                    "supported_labels": self.labels,
                    "ood_method": "disabled",
                    "checkpoint": self.config["checkpoint"],
                    "input_policy": "truncate to configured token budget",
                },
            )
            for text, score in zip(prepared, scores, strict=True)
        ]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        if target.exists() and any(target.iterdir()):
            raise ValueError("Refusing to overwrite an existing Laya artifact.")
        # The SDK normalizes tokenizer metadata on first load; hash its final form.
        self.client.initialize()
        source = self.client.prepare()
        if any(not (source / name).is_file() for name in CHECKPOINT_FILES):
            raise ValueError("Incomplete Laya checkpoint.")
        target.mkdir(parents=True, exist_ok=True)
        for name in CHECKPOINT_FILES:
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, destination)
        payload = {
            "artifact_version": 2,
            "model": "laya",
            "config": self.config,
            "taxonomy": self.taxonomy_identity,
            "labels": self.labels,
            "questions": self.questions,
            "metadata": self.metadata,
            "checkpoint_sha256": {
                name: compute_file_sha256(target / name) for name in CHECKPOINT_FILES
            },
        }
        (target / "naltra.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    def load(self, path: str | Path) -> None:
        target = Path(path)
        payload = json.loads((target / "naltra.json").read_text(encoding="utf-8"))
        if (
            payload["artifact_version"] != 2
            or payload["model"] != "laya"
            or payload["taxonomy"] != self.taxonomy_identity
            or set(payload["checkpoint_sha256"]) != set(CHECKPOINT_FILES)
            or any(
                compute_file_sha256(target / name) != payload["checkpoint_sha256"][name]
                for name in CHECKPOINT_FILES
            )
        ):
            raise ValueError("Incompatible or corrupted Laya artifact/taxonomy.")
        runtime_device = self.config["device"]
        self.__init__({**payload["config"], "device": runtime_device})
        if self.labels != payload["labels"] or self.questions != payload["questions"]:
            raise ValueError("Laya artifact topic questions or labels differ from the taxonomy.")
        self.metadata = payload["metadata"]
        self.client = LayaClient({**self.config, "device": str(self.device)}, checkpoint_dir=target)
