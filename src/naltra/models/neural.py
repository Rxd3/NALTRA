"""Shared local neural training, inference and artifact contract."""

from __future__ import annotations

import copy
import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from naltra.data.manifest import REPO_ROOT, compute_file_sha256, get_environment_metadata
from naltra.data.preprocessing import compute_content_fingerprint, validate_record
from naltra.models.base import BaseNALTRAModel
from naltra.pipeline.preprocessing import normalize_text
from naltra.pipeline.thresholds import apply_thresholds
from naltra.schemas.prediction import LanguageInfo, PredictionResult
from naltra.utils.config import load_yaml


def merge_config(defaults: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Merge nested configuration without modifying caller-owned mappings."""
    result = copy.deepcopy(dict(defaults))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def select_device(name: str) -> torch.device:
    if name not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda.")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu.")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate_neural_config(config: Mapping[str, Any]) -> None:
    training = config["training"]
    for key in ("batch_size", "epochs", "gradient_accumulation_steps", "patience"):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer.")
    for key in ("learning_rate", "gradient_clip"):
        if not math.isfinite(training[key]) or training[key] <= 0:
            raise ValueError(f"training.{key} must be finite and positive.")
    if not math.isfinite(training["weight_decay"]) or training["weight_decay"] < 0:
        raise ValueError("training.weight_decay must be finite and nonnegative.")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    if config["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda.")
    if (
        type(config["architecture"]["max_length"]) is not int
        or config["architecture"]["max_length"] < 1
    ):
        raise ValueError("architecture.max_length must be a positive integer.")
    apply_thresholds({}, config["multilabel"]["threshold"], config["multilabel"]["per_label"])


class EncodedRecords(Dataset):
    def __init__(self, encoded: dict[str, torch.Tensor], targets: torch.Tensor) -> None:
        self.encoded = encoded
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            **{key: value[index] for key, value in self.encoded.items()},
            "targets": self.targets[index],
        }


class NeuralModel(BaseNALTRAModel):
    """Subclasses supply initialization, encoding and network persistence."""

    model_name: str

    def __init__(
        self, config: Mapping[str, Any] | None = None, *, taxonomy_path: str | Path | None = None
    ) -> None:
        defaults = load_yaml(REPO_ROOT / "configs/models" / f"{self.model_name}.yaml")
        defaults = merge_config(
            {
                "device": "auto",
                "training": {
                    "gradient_accumulation_steps": 1,
                    "gradient_clip": 1.0,
                    "patience": 2,
                    "weight_decay": 0.01,
                    "mixed_precision": True,
                },
                "multilabel": {"per_label": {}},
            },
            defaults,
        )
        self.config = merge_config(defaults, config or {})
        validate_neural_config(self.config)
        if self.config["model"] != self.model_name:
            raise ValueError("Configuration model family does not match this model.")
        for key in ("enabled",):
            if type(self.config[key]) is not bool:
                raise ValueError(f"{key} must be a boolean.")
        for key in ("mixed_precision", "gradient_checkpointing"):
            if key in self.config["training"] and type(self.config["training"][key]) is not bool:
                raise ValueError(f"training.{key} must be a boolean.")
        self.taxonomy_path = Path(taxonomy_path or REPO_ROOT / "taxonomy/taxonomy.json")
        self.taxonomy = json.loads(self.taxonomy_path.read_text(encoding="utf-8"))
        self.taxonomy_identity = self._taxonomy_checksums()
        self.parents = {item["id"]: item["parent"] for item in self.taxonomy["labels"]}
        if set(self.config["multilabel"]["per_label"]) - set(self.parents):
            raise ValueError("Per-label thresholds must use canonical labels.")
        self.device = select_device(self.config["device"])
        self.labels: list[str] = []
        self.network: nn.Module | None = None
        self.history: list[dict[str, float | int | None]] = []
        self.metadata: dict[str, Any] = {}
        self.ood_threshold: float | None = None
        self._fitted = False

    def _taxonomy_checksums(self) -> dict[str, str]:
        taxonomy = json.loads(self.taxonomy_path.read_text(encoding="utf-8"))
        return {
            "taxonomy_version": taxonomy["version"],
            "taxonomy_sha256": compute_file_sha256(self.taxonomy_path),
            "label_map_sha256": compute_file_sha256(self.taxonomy_path.parent / "label_map.json"),
        }

    def _initialize(self, texts: Sequence[str]) -> None:
        raise NotImplementedError

    def _validate_architecture(self) -> None:
        """Validate subclass parameters both on construction and artifact loading."""

    def _encode(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def _logits(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def _save_network(self, target: Path) -> None:
        raise NotImplementedError

    def _load_network(self, target: Path) -> None:
        raise NotImplementedError

    def _records(self, data: Any, split: str) -> list[dict[str, Any]]:
        records = list(data)
        if not records:
            raise ValueError(f"{split} data cannot be empty.")
        for record in records:
            validate_record(record, allowed_labels=set(self.parents))
            if record["split"] != split:
                raise ValueError(
                    f"Expected {split} records; refusing to train/select on other splits."
                )
        if len({record["id"] for record in records}) != len(records):
            raise ValueError(f"Duplicate record IDs in {split} data.")
        return records

    def _loader(self, records: list[dict[str, Any]], *, shuffle: bool = False) -> DataLoader:
        index = {label: i for i, label in enumerate(self.labels)}
        targets = torch.zeros(len(records), len(index), dtype=torch.float32)
        for row, record in enumerate(records):
            for label in record["labels"]:
                if label not in index:
                    raise ValueError(
                        f"Validation label {label!r} is absent from training label space."
                    )
                targets[row, index[label]] = 1.0
        return DataLoader(
            EncodedRecords(self._encode([normalize_text(r["text"]) for r in records]), targets),
            batch_size=self.config["training"]["batch_size"],
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(self.config["seed"]),
            num_workers=0,
        )

    def train(self, train_data: Any, validation_data: Any | None = None) -> None:
        if self._taxonomy_checksums() != self.taxonomy_identity:
            raise ValueError("Taxonomy changed; construct a new model before training.")
        train = self._records(train_data, "train")
        validation = (
            self._records(validation_data, "validation") if validation_data is not None else None
        )
        if validation and {r["id"] for r in train} & {r["id"] for r in validation}:
            raise ValueError("Training and validation record IDs overlap.")
        if validation and {compute_content_fingerprint(r["text"]) for r in train} & {
            compute_content_fingerprint(r["text"]) for r in validation
        }:
            raise ValueError("Training and validation content overlaps.")
        seed_everything(self.config["seed"])
        self._fitted = False
        self.history = []
        self.ood_threshold = None
        self.labels = sorted({label for record in train for label in record["labels"]})
        unknown = {label for record in validation or [] for label in record["labels"]} - set(
            self.labels
        )
        if unknown:
            raise ValueError(
                f"Validation labels {sorted(unknown)} absent from training label space."
            )
        self._initialize([normalize_text(record["text"]) for record in train])
        assert self.network is not None
        self.network.to(self.device)
        loader = self._loader(train, shuffle=True)
        validation_loader = self._loader(validation) if validation else None
        cfg = self.config["training"]
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
            foreach=False,
        )
        criterion = nn.BCEWithLogitsLoss()
        use_amp = self.device.type == "cuda" and cfg["mixed_precision"]
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        best_loss, stale = float("inf"), 0
        best_state = None
        for epoch in range(cfg["epochs"]):
            self.network.train()
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for step, batch in enumerate(loader):
                targets = batch.pop("targets").to(self.device)
                inputs = {key: tensor.to(self.device) for key, tensor in batch.items()}
                with torch.autocast(self.device.type, dtype=torch.float16, enabled=use_amp):
                    loss = criterion(self._logits(inputs), targets)
                if not torch.isfinite(loss):
                    raise RuntimeError("Training produced non-finite loss.")
                total += loss.item() * len(targets)
                accumulation = cfg["gradient_accumulation_steps"]
                window_size = min(accumulation, len(loader) - (step // accumulation) * accumulation)
                scaler.scale(loss / window_size).backward()
                if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.network.parameters(), cfg["gradient_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            val_loss = self._validation_loss(validation_loader) if validation_loader else None
            self.history.append(
                {"epoch": epoch + 1, "train_loss": total / len(train), "validation_loss": val_loss}
            )
            if val_loss is not None:
                if val_loss < best_loss:
                    best_loss, stale = val_loss, 0
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.network.state_dict().items()
                    }
                else:
                    stale += 1
                if stale >= cfg["patience"]:
                    break
        if best_state is not None:
            self.network.load_state_dict(best_state)
        self.network.eval()
        self._fitted = True
        if validation:
            scores = self._probabilities([normalize_text(r["text"]) for r in validation])
            self.ood_threshold = float(np.quantile(1.0 - scores.max(axis=1), 0.95))
        self.metadata.update(
            {
                "train_count": len(train),
                "validation_count": len(validation or []),
                "environment": get_environment_metadata(),
                "device": str(self.device),
                "hardware": (
                    torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
                ),
            }
        )

    def _validation_loss(self, loader: DataLoader) -> float:
        assert self.network is not None
        self.network.eval()
        total = 0.0
        with torch.inference_mode():
            for batch in loader:
                targets = batch.pop("targets").to(self.device)
                logits = self._logits({key: value.to(self.device) for key, value in batch.items()})
                loss = nn.functional.binary_cross_entropy_with_logits(logits.float(), targets)
                if not torch.isfinite(loss):
                    raise RuntimeError("Validation produced non-finite loss.")
                total += loss.item() * len(targets)
        return total / len(loader.dataset)

    def _probabilities(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted or self.network is None:
            raise RuntimeError(f"{self.model_name} must be trained or loaded before prediction.")
        self.network.eval()
        chunks = []
        with torch.inference_mode():
            size = self.config["training"]["batch_size"]
            for start in range(0, len(texts), size):
                inputs = {
                    key: value.to(self.device)
                    for key, value in self._encode(texts[start : start + size]).items()
                }
                chunks.append(torch.sigmoid(self._logits(inputs).float()).cpu().numpy())
        return np.concatenate(chunks) if chunks else np.empty((0, len(self.labels)))

    def predict(self, text: str) -> PredictionResult:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: Iterable[str]) -> list[PredictionResult]:
        prepared = [normalize_text(text) for text in texts]
        if any(not text for text in prepared):
            raise ValueError("Prediction texts cannot contain empty values.")
        start = perf_counter()
        probabilities = self._probabilities(prepared)
        elapsed = (perf_counter() - start) * 1000 / max(1, len(prepared))
        results = []
        for text, row in zip(prepared, probabilities, strict=True):
            scores = dict(zip(self.labels, map(float, row), strict=True))
            results.append(
                PredictionResult(
                    text=text,
                    model=self.model_name,
                    language=LanguageInfo(primary="und"),
                    labels=apply_thresholds(
                        scores,
                        self.config["multilabel"]["threshold"],
                        self.config["multilabel"]["per_label"],
                    ),
                    label_scores=scores,
                    latency_ms=elapsed,
                    metadata={
                        "supported_labels": self.labels.copy(),
                        "ood_threshold": self.ood_threshold,
                        "ood_method": (
                            "max_probability" if self.ood_threshold is not None else "disabled"
                        ),
                        "calibration": "none",
                        "taxonomy_version": self.taxonomy["version"],
                        **{
                            key: self.metadata[key]
                            for key in ("dataset", "track")
                            if key in self.metadata
                        },
                    },
                )
            )
        return results

    def save(self, path: str | Path) -> None:
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        if self._taxonomy_checksums() != self.taxonomy_identity:
            raise ValueError("Taxonomy changed since initialization; refusing a stale artifact.")
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        self._save_network(target)
        payload = {
            "artifact_version": 1,
            "model": self.model_name,
            "config": self.config,
            "labels": self.labels,
            "taxonomy": self.taxonomy_identity,
            "ood_threshold": self.ood_threshold,
            "history": self.history,
            "metadata": self.metadata,
        }
        (target / "naltra.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    def load(self, path: str | Path) -> None:
        self._fitted = False
        target = Path(path)
        payload = json.loads((target / "naltra.json").read_text(encoding="utf-8"))
        if payload["artifact_version"] != 1 or payload["model"] != self.model_name:
            raise ValueError("Incompatible model artifact.")
        if (
            payload["taxonomy"] != self.taxonomy_identity
            or self._taxonomy_checksums() != self.taxonomy_identity
        ):
            raise ValueError("Artifact taxonomy version or hashes do not match.")
        labels = payload["labels"]
        if not labels or len(set(labels)) != len(labels) or not set(labels) <= set(self.parents):
            raise ValueError("Invalid artifact label space.")
        # The caller's selected device takes precedence over the producing machine.
        self.config = merge_config(payload["config"], {"device": self.config["device"]})
        validate_neural_config(self.config)
        self._validate_architecture()
        self.device = select_device(self.config["device"])
        self.labels, self.history, self.metadata = labels, payload["history"], payload["metadata"]
        self.ood_threshold = payload["ood_threshold"]
        if self.ood_threshold is not None and not 0 <= self.ood_threshold <= 1:
            raise ValueError("Invalid artifact OOD threshold.")
        self._fitted = False
        self._load_network(target)
        assert self.network is not None
        self.network.to(self.device).eval()
        self._fitted = True
