"""Durable, provenance-bound predictions for resumable evaluation runs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from naltra.pipeline.preprocessing import normalize_text
from naltra.pipeline.thresholds import apply_thresholds
from naltra.schemas.prediction import LanguageInfo, PredictionResult


class PredictionCache:
    """Commit whole batches atomically and reject changed inputs or model settings."""

    def __init__(self, path: Path, key: str, identity: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.key = key
        try:
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS runs (key TEXT PRIMARY KEY, identity TEXT NOT NULL)"
            )
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS predictions ("
                "run TEXT NOT NULL, position INTEGER NOT NULL, payload TEXT NOT NULL, "
                "sha256 TEXT NOT NULL, PRIMARY KEY (run, position))"
            )
            encoded = json.dumps(identity, sort_keys=True, allow_nan=False)
            previous = self.connection.execute(
                "SELECT identity FROM runs WHERE key = ?", (key,)
            ).fetchone()
            if previous is not None and previous[0] != encoded:
                raise ValueError("Prediction cache provenance differs; use a new output directory.")
            with self.connection:
                self.connection.execute("INSERT OR IGNORE INTO runs VALUES (?, ?)", (key, encoded))
        except BaseException:
            self.connection.close()
            raise

    def load(self, records: Sequence[Mapping[str, Any]], model: Any) -> list[PredictionResult]:
        predictions = []
        for position, payload, digest in self.connection.execute(
            "SELECT position, payload, sha256 FROM predictions WHERE run = ? ORDER BY position",
            (self.key,),
        ):
            if position != len(predictions) or position >= len(records):
                raise ValueError("Prediction cache is not an aligned contiguous prefix.")
            if hashlib.sha256(payload.encode()).hexdigest() != digest:
                raise ValueError("Prediction cache checksum differs.")
            saved = json.loads(payload)
            if (
                saved["text"] != normalize_text(records[position]["text"])
                or saved["model"] != model.model_name
                or set(saved["scores"]) != set(model.labels)
            ):
                raise ValueError("Prediction cache text, model or labels differ.")
            predictions.append(
                PredictionResult(
                    text=saved["text"],
                    model=saved["model"],
                    language=LanguageInfo("und"),
                    label_scores=saved["scores"],
                    labels=apply_thresholds(
                        saved["scores"],
                        model.config["multilabel"]["threshold"],
                        model.config["multilabel"]["per_label"],
                    ),
                    latency_ms=saved["latency_ms"],
                )
            )
        return predictions

    def append(self, offset: int, predictions: Sequence[PredictionResult]) -> None:
        rows = []
        for position, prediction in enumerate(predictions, offset):
            payload = json.dumps(
                {
                    "text": prediction.text,
                    "model": prediction.model,
                    "scores": prediction.label_scores,
                    "latency_ms": prediction.latency_ms,
                },
                allow_nan=False,
            )
            rows.append((self.key, position, payload, hashlib.sha256(payload.encode()).hexdigest()))
        with self.connection:
            self.connection.executemany("INSERT INTO predictions VALUES (?, ?, ?, ?)", rows)

    def close(self) -> None:
        self.connection.close()
