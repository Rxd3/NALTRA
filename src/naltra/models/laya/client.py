"""Local, bounded-memory inference using the optional Laya SDK."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CHECKPOINT_FILES = (
    "rl_agent_config.json",
    "model.safetensors",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)


class LayaClient:
    def __init__(self, config: Mapping[str, Any], *, checkpoint_dir: Path | None = None) -> None:
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.agent: Any = None

    def prepare(self) -> Path:
        if self.checkpoint_dir is None:
            from huggingface_hub import snapshot_download

            checkpoint = self.config["checkpoint"]
            prefix = checkpoint["subfolder"] + "/"
            snapshot = snapshot_download(
                checkpoint["repo_id"],
                revision=checkpoint["revision"],
                allow_patterns=[prefix + name for name in CHECKPOINT_FILES],
            )
            self.checkpoint_dir = Path(snapshot) / checkpoint["subfolder"]
        return self.checkpoint_dir

    def initialize(self) -> None:
        if self.agent is not None:
            return
        try:
            import laya
        except ImportError as exc:
            raise RuntimeError(
                "Install local Laya with: python -m pip install -r requirements-laya.txt"
            ) from exc
        if laya.__version__ != self.config["sdk_version"]:
            raise RuntimeError(f"This artifact requires laya=={self.config['sdk_version']}.")
        agent = laya.load(str(self.prepare()), device=self.config["device"], backend="eager")
        if self.config["device"] == "cuda" and agent.device.type != "cuda":
            raise RuntimeError("Laya could not load on CUDA; reduce batching or select CPU.")
        self.agent = agent

    def predict_batch(
        self, texts: list[str], questions: Mapping[str, dict[str, Any]]
    ) -> list[dict[str, float]]:
        if not texts:
            return []
        self.initialize()
        results: list[dict[str, float]] = [{} for _ in texts]
        items = list(questions.items())
        inference = self.config["inference"]
        for offset in range(0, len(items), inference["question_batch_size"]):
            group = dict(items[offset : offset + inference["question_batch_size"]])
            answers = self.agent.predict_batch(
                texts,
                group,
                batch_size=inference["batch_size"],
                max_len=inference["max_length"],
                head_max_len=inference["head_max_length"],
            )
            if len(answers) != len(texts):
                raise ValueError("Laya returned an incorrect batch length.")
            for scores, response in zip(results, answers, strict=True):
                if (
                    not isinstance(response, Mapping)
                    or not isinstance(response.get("answers"), Mapping)
                    or set(response["answers"]) != set(group)
                ):
                    raise ValueError("Laya must answer every requested topic exactly once.")
                for label, answer in response["answers"].items():
                    if not isinstance(answer, Mapping):
                        raise ValueError("Invalid Laya topic answer.")
                    probabilities = answer.get("probabilities", {})
                    if answer.get("type") != "choice" or set(probabilities) != {"A", "B"}:
                        raise ValueError("Laya must return two-option topic probabilities.")
                    values = list(probabilities.values())
                    if any(
                        isinstance(p, bool)
                        or not isinstance(p, (int, float))
                        or not math.isfinite(p)
                        or not 0 <= p <= 1
                        for p in values
                    ) or not math.isclose(sum(values), 1, abs_tol=0.001):
                        raise ValueError("Invalid Laya topic probabilities.")
                    scores[label] = float(probabilities["A"])
        return results
