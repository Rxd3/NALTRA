"""Shared TF-IDF + one-vs-rest training, inference and artifact contract.

After fitting, the per-label scikit-learn estimators are collapsed into stacked weight
matrices, so scoring is one sparse matrix product instead of one Python call per label. A
self-check refuses the compact form unless it reproduces scikit-learn's probabilities.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import sklearn
from scipy.sparse import spmatrix
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from naltra.data.manifest import (
    REPO_ROOT,
    compute_file_sha256,
    get_environment_metadata,
    get_taxonomy_checksums,
)
from naltra.data.preprocessing import validate_record
from naltra.data.validation import check_disjoint_partitions
from naltra.models.base import BaseNALTRAModel
from naltra.pipeline.preprocessing import normalize_text
from naltra.pipeline.thresholds import apply_thresholds
from naltra.schemas.prediction import LanguageInfo, PredictionResult
from naltra.utils.config import load_yaml, merge_config

ESTIMATOR_FILE = "estimator.joblib"
SELF_CHECK_ROWS = 64
ARTIFACT_VERSION = 2
COEF_LAYOUT = "feature_major"


def turkish_safe_lower(text: str) -> str:
    """Lowercase without str.lower turning Turkish 'İ' into 'i' plus a combining dot."""
    return text.replace("İ", "i").lower()


def translation_group(record: Mapping[str, Any]) -> str:
    """Key shared by a project's EN source, TR translation and code-switched variants."""
    return record.get("pair_id") or record.get("project_id") or record["id"]


def _is_positive_int(value: Any) -> bool:
    return type(value) is int and value >= 1


class ClassicalModel(BaseNALTRAModel):
    """Subclasses supply the per-label binary classifier and its compact scoring form."""

    model_name: str
    calibration = "none"

    def __init__(
        self, config: Mapping[str, Any] | None = None, *, taxonomy_path: str | Path | None = None
    ) -> None:
        defaults = merge_config(
            {"target_field": "labels", "seed": 42, "n_jobs": -1, "multilabel": {"per_label": {}}},
            load_yaml(REPO_ROOT / "configs/models" / f"{self.model_name}.yaml"),
        )
        self.config = merge_config(defaults, config or {})
        taxonomy_file = Path(taxonomy_path or REPO_ROOT / "taxonomy/taxonomy.json")
        self.taxonomy_dir = taxonomy_file.parent
        self.taxonomy = json.loads(taxonomy_file.read_text(encoding="utf-8"))
        self.taxonomy_identity = get_taxonomy_checksums(self.taxonomy_dir)
        self.parents = {item["id"]: item["parent"] for item in self.taxonomy["labels"]}
        self._validate_config()
        self.labels: list[str] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.weights: dict[str, np.ndarray] = {}
        self.metadata: dict[str, Any] = {}
        self.ood_threshold: float | None = None
        self._fitted = False

    def _validate_config(self) -> None:
        config = self.config
        if config["model"] != self.model_name:
            raise ValueError("Configuration model family does not match this model.")
        if type(config["enabled"]) is not bool:
            raise ValueError("enabled must be a boolean.")
        if config["target_field"] not in {"labels", "labels_direct"}:
            raise ValueError("target_field must be labels or labels_direct.")
        if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")
        if type(config["n_jobs"]) is not int or config["n_jobs"] == 0:
            raise ValueError("n_jobs must be a nonzero integer.")
        features = config["features"]
        low, high = features["ngram_range"]
        if not (_is_positive_int(low) and _is_positive_int(high) and low <= high):
            raise ValueError("features.ngram_range must be two ordered positive integers.")
        if features.get("max_features") is not None and not _is_positive_int(
            features["max_features"]
        ):
            raise ValueError("features.max_features must be a positive integer or null.")
        if not _is_positive_int(features.get("min_df", 1)):
            raise ValueError("features.min_df must be a positive integer.")
        multilabel = config["multilabel"]
        if not isinstance(multilabel, dict) or set(multilabel) - {"threshold", "per_label"}:
            raise ValueError("multilabel must hold only threshold and per_label.")
        per_label = multilabel.get("per_label")
        if not isinstance(per_label, dict) or not all(isinstance(key, str) for key in per_label):
            raise ValueError("multilabel.per_label must map label names to thresholds.")
        apply_thresholds({}, multilabel["threshold"], multilabel["per_label"])
        if set(multilabel["per_label"]) - set(self.parents):
            raise ValueError("Per-label thresholds must use canonical labels.")
        self._validate_classifier()

    def _validate_classifier(self) -> None:
        """Validate subclass classifier parameters on construction and loading."""

    def _classifier(self, groups: Sequence[str]) -> ClassifierMixin:
        """Per-label estimator; internal folds must keep each translation group together."""
        raise NotImplementedError

    def _check_trainable(self, targets: np.ndarray, groups: Sequence[str]) -> None:
        """Reject label matrices the subclass classifier cannot fit."""

    def _compact(self, estimators: Sequence[Any]) -> dict[str, np.ndarray]:
        """Stack fitted per-label estimators; ``coef`` is feature-major (labels on last axis)."""
        raise NotImplementedError

    def _score_matrix(self, features: spmatrix, texts: Sequence[str]) -> np.ndarray:
        """Positive-class probabilities computed from the fitted weights."""
        raise NotImplementedError

    def _fit(
        self, features: spmatrix, targets: np.ndarray, texts: list[str], groups: Sequence[str]
    ) -> dict:
        """Fit one-vs-rest estimators and return their self-checked compact weights."""
        classifier = OneVsRestClassifier(self._classifier(groups), n_jobs=self.config["n_jobs"])
        classifier.fit(features, targets)
        self.weights = self._compact(classifier.estimators_)
        rows = slice(0, SELF_CHECK_ROWS)
        compact = self._score_matrix(features[rows], texts[rows])
        if not np.allclose(compact, classifier.predict_proba(features[rows]), atol=1e-4):
            raise RuntimeError("Compact scoring diverged from scikit-learn; check its version.")
        return self.weights

    def _check_weights(self, weights: dict, label_count: int) -> None:
        if weights["coef"].shape[-1] != label_count:
            raise ValueError("Estimator and artifact label space disagree.")

    def _vectorizer(self) -> TfidfVectorizer:
        features = self.config["features"]
        return TfidfVectorizer(
            preprocessor=turkish_safe_lower if features["lowercase"] else None,
            lowercase=features["lowercase"],
            ngram_range=tuple(features["ngram_range"]),
            max_features=features.get("max_features"),
            min_df=features.get("min_df", 1),
            sublinear_tf=features.get("sublinear_tf", True),
            dtype=np.float32,
        )

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
            if self.config["target_field"] not in record:
                raise ValueError(f"Record {record['id']!r} lacks the configured target field.")
        if len({record["id"] for record in records}) != len(records):
            raise ValueError(f"Duplicate record IDs in {split} data.")
        return records

    def _label_matrix(self, train: list[dict], validation: list[dict]) -> tuple:
        field = self.config["target_field"]
        binarizer = MultiLabelBinarizer()
        targets = binarizer.fit_transform([record[field] for record in train])
        labels = [str(label) for label in binarizer.classes_]
        if len(labels) < 2:
            raise ValueError("Multi-label training needs at least two distinct labels.")
        unknown = {label for record in validation for label in record[field]} - set(labels)
        if unknown:
            raise ValueError(f"Validation labels {sorted(unknown)} absent from training labels.")
        constant = [label for label, column in zip(labels, targets.T, strict=True) if column.all()]
        if constant:
            raise ValueError(f"Labels {constant} occur in every training record; nothing to learn.")
        return labels, targets

    def train(self, train_data: Any, validation_data: Any | None = None) -> None:
        if get_taxonomy_checksums(self.taxonomy_dir) != self.taxonomy_identity:
            raise ValueError("Taxonomy changed; construct a new model before training.")
        train = self._records(train_data, "train")
        validation = (
            self._records(validation_data, "validation") if validation_data is not None else []
        )
        check_disjoint_partitions({"train": train, "validation": validation})
        self._fitted, self.ood_threshold = False, None
        labels, targets = self._label_matrix(train, validation)
        groups = [translation_group(record) for record in train]
        self._check_trainable(targets, groups)
        texts = [normalize_text(record["text"]) for record in train]
        vectorizer = self._vectorizer()
        features = vectorizer.fit_transform(texts)
        self.labels, self.vectorizer = labels, vectorizer
        self.weights = self._fit(features, targets, texts, groups)
        self._fitted = True
        if validation:
            scores = self._probabilities([normalize_text(r["text"]) for r in validation])
            self.ood_threshold = float(np.quantile(1.0 - scores.max(axis=1), 0.95))
        self.metadata.update(
            {
                "train_count": len(train),
                "validation_count": len(validation),
                "environment": get_environment_metadata(),
                "device": "cpu",
                "hardware": "CPU",
            }
        )

    def _probabilities(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted or self.vectorizer is None:
            raise RuntimeError(f"{self.model_name} must be trained or loaded before prediction.")
        if not texts:
            return np.empty((0, len(self.labels)))
        # Clip float round-off so every score passes the [0, 1] schema check.
        texts = list(texts)
        return np.clip(self._score_matrix(self.vectorizer.transform(texts), texts), 0.0, 1.0)

    def predict(self, text: str) -> PredictionResult:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: Iterable[str]) -> list[PredictionResult]:
        prepared = [normalize_text(text) for text in texts]
        if any(not text for text in prepared):
            raise ValueError("Prediction texts cannot contain empty values.")
        start = perf_counter()
        probabilities = self._probabilities(prepared)
        elapsed = (perf_counter() - start) * 1000 / max(1, len(prepared))
        multilabel = self.config["multilabel"]
        metadata = {
            "ood_threshold": self.ood_threshold,
            "ood_method": "max_probability" if self.ood_threshold is not None else "disabled",
            "calibration": self.calibration,
            "taxonomy_version": self.taxonomy["version"],
            **{key: self.metadata[key] for key in ("dataset", "track") if key in self.metadata},
        }
        results = []
        for text, row in zip(prepared, probabilities, strict=True):
            scores = dict(zip(self.labels, map(float, row), strict=True))
            results.append(
                PredictionResult(
                    text=text,
                    model=self.model_name,
                    language=LanguageInfo(primary="und"),
                    labels=apply_thresholds(
                        scores, multilabel["threshold"], multilabel["per_label"]
                    ),
                    label_scores=scores,
                    latency_ms=elapsed,
                    metadata={**metadata, "supported_labels": self.labels.copy()},
                )
            )
        return results

    def save(self, path: str | Path) -> None:
        if not self._fitted or self.vectorizer is None:
            raise RuntimeError("Cannot save an unfitted model.")
        if get_taxonomy_checksums(self.taxonomy_dir) != self.taxonomy_identity:
            raise ValueError("Taxonomy changed since initialization; refusing a stale artifact.")
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"vectorizer": self.vectorizer, "weights": self.weights}, target / ESTIMATOR_FILE
        )
        payload = {
            "artifact_version": ARTIFACT_VERSION,
            "coef_layout": COEF_LAYOUT,
            "model": self.model_name,
            "sklearn_version": sklearn.__version__,
            "config": self.config,
            "labels": self.labels,
            "taxonomy": self.taxonomy_identity,
            "ood_threshold": self.ood_threshold,
            "metadata": self.metadata,
            "estimator_sha256": compute_file_sha256(target / ESTIMATOR_FILE),
        }
        (target / "naltra.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    def _check_payload(self, payload: dict[str, Any]) -> None:
        if payload.get("artifact_version") == 1:
            raise ValueError(
                "Artifact version 1 stores label-major coefficients from before the "
                "feature-major change; retrain the model."
            )
        if (
            payload.get("artifact_version") != ARTIFACT_VERSION
            or payload.get("coef_layout") != COEF_LAYOUT
            or payload["model"] != self.model_name
        ):
            raise ValueError("Incompatible model artifact.")
        if payload.get("sklearn_version") != sklearn.__version__:
            raise ValueError(
                f"Artifact was pickled by scikit-learn {payload.get('sklearn_version')}, but "
                f"{sklearn.__version__} is installed; retrain or install the matching version."
            )
        if (
            payload["taxonomy"] != self.taxonomy_identity
            or get_taxonomy_checksums(self.taxonomy_dir) != self.taxonomy_identity
        ):
            raise ValueError("Artifact taxonomy version or hashes do not match.")
        labels = payload["labels"]
        if (
            len(labels) < 2
            or len(set(labels)) != len(labels)
            or not set(labels) <= set(self.parents)
        ):
            raise ValueError("Invalid artifact label space.")
        threshold = payload["ood_threshold"]
        if threshold is not None and not (math.isfinite(threshold) and 0 <= threshold <= 1):
            raise ValueError("Invalid artifact OOD threshold.")

    def load(self, path: str | Path) -> None:
        self._fitted = False
        target = Path(path)
        payload = json.loads((target / "naltra.json").read_text(encoding="utf-8"))
        self._check_payload(payload)
        # joblib unpickles code: the checksum only guards our own artifacts against corruption.
        if compute_file_sha256(target / ESTIMATOR_FILE) != payload["estimator_sha256"]:
            raise ValueError("Estimator checksum mismatch; refusing to load the artifact.")
        # The device is the caller's runtime choice, never the producing machine's.
        config = {key: value for key, value in payload["config"].items() if key != "device"}
        if "device" in self.config:
            config["device"] = self.config["device"]
        self.config = config
        self._validate_config()
        state = joblib.load(target / ESTIMATOR_FILE)
        labels = payload["labels"]
        self._check_weights(state["weights"], len(labels))
        self.labels, self.metadata = labels, payload["metadata"]
        self.ood_threshold = payload["ood_threshold"]
        self.vectorizer, self.weights, self._fitted = state["vectorizer"], state["weights"], True
