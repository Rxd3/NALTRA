"""Explicit JSON service contracts; no provider endpoint or score format is assumed."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, ClassVar

import httpx
from dotenv import load_dotenv

from naltra.data.manifest import REPO_ROOT, get_taxonomy_checksums
from naltra.models.base import BaseNALTRAModel
from naltra.models.neural import merge_config
from naltra.pipeline.preprocessing import normalize_text
from naltra.pipeline.thresholds import apply_thresholds
from naltra.schemas.prediction import LanguageInfo, PredictionResult
from naltra.utils.config import load_yaml


@dataclass(slots=True)
class JSONServiceClient:
    api_key: str = field(repr=False)
    base_url: str
    request_path: str = ""
    text_field: str = "text"
    scores_path: str = "scores"
    auth_header: str = "Authorization"
    auth_prefix: str = "Bearer "
    timeout: float = 30.0
    configured: bool = False
    transport: httpx.BaseTransport | None = field(default=None, repr=False)
    service_name: ClassVar[str]

    def __post_init__(self) -> None:
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Service timeout must be finite and positive.")
        if type(self.configured) is not bool:
            raise ValueError("contract.configured must be a boolean.")
        if not self.text_field or not self.scores_path or not self.auth_header:
            raise ValueError("Service contract field names cannot be empty.")

    @classmethod
    def from_environment(cls, config: Mapping[str, Any] | None = None) -> JSONServiceClient:
        config = config or load_yaml(REPO_ROOT / "configs/models" / f"{cls.service_name}.yaml")
        load_dotenv(REPO_ROOT / ".env", override=False)
        key = os.getenv(config["api_key_env"], "")
        url = os.getenv(config["base_url_env"], "")
        if not key or not url:
            raise RuntimeError(
                f"{config['api_key_env']} and {config['base_url_env']} "
                "must be set in the environment."
            )
        return cls(
            api_key=key,
            base_url=url,
            timeout=config["request_timeout_seconds"],
            **config["contract"],
        )

    def predict(self, text: str) -> dict[str, Any]:
        text = normalize_text(text)
        if not text:
            raise ValueError("Service prediction text cannot be empty.")
        if not self.configured:
            raise RuntimeError(
                f"{self.service_name} provider contract is not configured; see docs/core_ml.md."
            )
        if not self.api_key or not self.base_url:
            raise RuntimeError("Service credentials and base URL are required.")
        url = httpx.URL(self.base_url)
        if url.scheme not in {"http", "https"} or not url.host:
            raise ValueError("Service base URL must be an HTTP(S) URL.")
        if self.request_path:
            if self.request_path.startswith(("http:", "https:", "//")):
                raise ValueError("request_path must be a relative provider path.")
            url = httpx.URL(str(url).rstrip("/") + "/" + self.request_path.lstrip("/"))
        try:
            with httpx.Client(
                timeout=self.timeout, transport=self.transport, follow_redirects=False
            ) as client:
                response = client.post(
                    url,
                    json={self.text_field: text},
                    headers={self.auth_header: self.auth_prefix + self.api_key},
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"{self.service_name} request failed ({type(exc).__name__})."
            ) from None
        except ValueError:
            raise ValueError(f"{self.service_name} response is not valid JSON.") from None
        for part in self.scores_path.split("."):
            if not isinstance(data, dict) or part not in data:
                raise ValueError("Service response does not match configured scores_path.")
            data = data[part]
        if not isinstance(data, dict) or not data:
            raise ValueError("Service scores must be a non-empty label-to-probability object.")
        for label, score in data.items():
            if (
                not isinstance(label, str)
                or not label.strip()
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
                or not 0 <= score <= 1
            ):
                raise ValueError("Service scores must contain finite probabilities in [0, 1].")
        return {"scores": data}


class ExternalServiceModel(BaseNALTRAModel):
    model_name: str
    client_type: type[JSONServiceClient]

    def __init__(
        self, client: JSONServiceClient | None = None, config: Mapping[str, Any] | None = None
    ) -> None:
        self.config = merge_config(
            load_yaml(REPO_ROOT / "configs/models" / f"{self.model_name}.yaml"), config or {}
        )
        self._validate_config()
        self.client = client

    def _validate_config(self) -> None:
        allowed = {
            "model",
            "enabled",
            "api_key_env",
            "base_url_env",
            "request_timeout_seconds",
            "contract",
            "multilabel",
            "supported_labels",
            "label_map",
        }
        if set(self.config) - allowed:
            raise ValueError(
                "Unknown service configuration keys; credentials must use environment variables."
            )
        if self.config["model"] != self.model_name or type(self.config["enabled"]) is not bool:
            raise ValueError("Service model name/enabled configuration is invalid.")
        for key in ("api_key_env", "base_url_env"):
            if not isinstance(self.config[key], str) or not self.config[key]:
                raise ValueError(f"{key} must name an environment variable.")
        timeout = self.config["request_timeout_seconds"]
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Service timeout must be finite and positive.")
        labels = self.config["supported_labels"]
        if (
            not isinstance(labels, list)
            or any(not isinstance(label, str) for label in labels)
            or len(labels) != len(set(labels))
        ):
            raise ValueError("Service supported_labels must be a list of unique labels.")
        taxonomy = json.loads((REPO_ROOT / "taxonomy/taxonomy.json").read_text(encoding="utf-8"))
        canonical = {item["id"] for item in taxonomy["labels"]}
        if not set(labels) <= canonical or not set(self.config["label_map"].values()) <= set(
            labels
        ):
            raise ValueError("Service label mapping must target supported canonical labels.")
        apply_thresholds(
            {},
            self.config["multilabel"]["threshold"],
            self.config["multilabel"].get("per_label", {}),
        )
        contract_keys = {
            "configured",
            "request_path",
            "text_field",
            "scores_path",
            "auth_header",
            "auth_prefix",
        }
        if set(self.config["contract"]) != contract_keys:
            raise ValueError("Invalid service contract fields.")
        self.client_type(api_key="", base_url="", timeout=timeout, **self.config["contract"])

    def train(self, train_data: Any, validation_data: Any | None = None) -> None:
        del train_data, validation_data
        raise RuntimeError(
            f"{self.model_name} is an inference service; local training is unsupported."
        )

    def predict(self, text: str) -> PredictionResult:
        if not self.config["enabled"]:
            raise RuntimeError(
                f"{self.model_name} is disabled; "
                "configure the real provider contract before enabling it."
            )
        if not self.config["supported_labels"]:
            raise RuntimeError("Configure the service supported_labels for the experiment.")
        text = normalize_text(text)
        if not text:
            raise ValueError("Prediction text cannot be empty.")
        if self.client is None:
            self.client = self.client_type.from_environment(self.config)
        start = perf_counter()
        data = self.client.predict(text)["scores"]
        scores = {}
        for source, probability in data.items():
            label = self.config["label_map"].get(source, source)
            if label not in self.config["supported_labels"]:
                raise ValueError(f"Unknown or unsupported service label: {label!r}.")
            if label in scores:
                raise ValueError("Multiple service labels map to the same canonical label.")
            scores[label] = probability
        if set(scores) != set(self.config["supported_labels"]):
            raise ValueError("Service must return a probability for every supported label.")
        return PredictionResult(
            text=text,
            model=self.model_name,
            language=LanguageInfo(primary="und"),
            labels=apply_thresholds(
                scores,
                self.config["multilabel"]["threshold"],
                self.config["multilabel"].get("per_label", {}),
            ),
            label_scores=scores,
            latency_ms=(perf_counter() - start) * 1000,
            metadata={
                "supported_labels": sorted(scores),
                "ood_method": "disabled",
                "calibration": "provider_unspecified",
            },
        )

    def predict_batch(self, texts: Iterable[str]) -> list[PredictionResult]:
        prepared = [normalize_text(text) for text in texts]
        if any(not text for text in prepared):
            raise ValueError("Prediction texts cannot contain empty values.")
        return [self.predict(text) for text in prepared]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifact_version": 1,
            "model": self.model_name,
            "config": self.config,
            "taxonomy": get_taxonomy_checksums(REPO_ROOT / "taxonomy"),
        }
        (target / "naltra.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    def load(self, path: str | Path) -> None:
        payload = json.loads((Path(path) / "naltra.json").read_text(encoding="utf-8"))
        if (
            payload["artifact_version"] != 1
            or payload["model"] != self.model_name
            or payload["taxonomy"] != get_taxonomy_checksums(REPO_ROOT / "taxonomy")
        ):
            raise ValueError("Incompatible service artifact or taxonomy.")
        self.config = payload["config"]
        self._validate_config()
        self.client = None
