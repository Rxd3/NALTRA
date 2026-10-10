"""Hybrid sparse + dense retrieval kNN: neighbours found by TF-IDF and multilingual-e5
cosine similarity, fused with reciprocal rank fusion, then vote with their labels.

Reciprocal rank fusion follows Cormack, Clarke and Buettcher (SIGIR 2009) with k = 60.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import spmatrix

from naltra.models.classical import ClassicalModel, _is_positive_int

Encoder = Callable[[Sequence[str]], np.ndarray]
QUERY_CHUNK = 256
# Fixed English and Turkish probes whose embeddings fingerprint the dense encoder.
CANARY_TEXTS = (
    "Acoustic sensors measure sound waves in concert halls.",
    "Lazer ışığı mercekten geçerek kırılır.",
)
# e5 in fp16/bf16 on CPU moved the canaries by at most 1.1e-3; the two canaries differ by 0.09.
CANARY_ATOL = 5e-3
# A full commit SHA; branch and tag names such as "main" can move.
COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


def reciprocal_rank_fusion(
    similarities: Sequence[np.ndarray], neighbors: int, k: int
) -> np.ndarray:
    """Sum 1/(k + rank) over each retriever's top-``neighbors`` documents (rank from 1).

    Only documents with positive similarity are candidates: a query sharing nothing with the
    index gets no votes, instead of ones decided by training-row order.
    """
    fused = np.zeros_like(similarities[0], dtype=np.float64)
    rows = np.arange(len(fused))[:, None]
    for similarity in similarities:
        top = np.argsort(-similarity, axis=1, kind="stable")[:, :neighbors]
        hit = np.take_along_axis(similarity, top, axis=1) > 0
        fused[rows, top] += hit / (k + np.arange(1, top.shape[1] + 1))
    return fused


class E5Encoder:
    """Mean-pooled, L2-normalised multilingual-e5 sentence embeddings, loaded lazily."""

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.settings = dict(settings)
        self.tokenizer = self.network = self.device = None

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        import torch
        from transformers import AutoModel, AutoTokenizer

        from naltra.models.neural import select_device

        if self.network is None:
            name, revision = self.settings["pretrained_name"], self.settings.get("revision")
            self.device = select_device(self.settings.get("device", "auto"))
            self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
            self.network = AutoModel.from_pretrained(name, revision=revision).to(self.device).eval()
        chunks = []
        size = self.settings.get("batch_size", 32)
        with torch.inference_mode():
            for start in range(0, len(texts), size):
                # e5 expects the "query: " prefix for symmetric document-to-document search.
                batch = self.tokenizer(
                    [f"query: {text}" for text in texts[start : start + size]],
                    truncation=True,
                    padding=True,
                    max_length=self.settings.get("max_length", 256),
                    return_tensors="pt",
                ).to(self.device)
                with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
                    hidden = self.network(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                chunks.append(torch.nn.functional.normalize(pooled.float(), dim=-1).cpu().numpy())
        return np.concatenate(chunks) if chunks else np.empty((0, 0), dtype=np.float32)


class HybridKNNModel(ClassicalModel):
    """Instance-based member: no learned weights, so it adds a distinct inductive bias."""

    model_name = "hybrid_knn"

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        taxonomy_path: str | Path | None = None,
        encoder: Encoder | None = None,
    ) -> None:
        super().__init__(config, taxonomy_path=taxonomy_path)
        self._injected_encoder = self._encoder = encoder
        self._encoder_verified = True

    @property
    def encoder(self) -> Encoder:
        if self._encoder is None:
            settings = self.config["encoder"]
            # The CLI-level --device wins over the encoder's own default.
            device = self.config.get("device", settings.get("device", "auto"))
            self._encoder = E5Encoder({**settings, "device": device})
        return self._encoder

    def load(self, path: str | Path) -> None:
        # A lazily built e5 encoder must follow the artifact's settings and the caller's device,
        # and must reproduce the artifact's canary embeddings before its first dense query.
        self._encoder = self._injected_encoder
        self._encoder_verified = False
        super().load(path)

    def _validate_classifier(self) -> None:
        retrieval = self.config["retrieval"]
        if not _is_positive_int(retrieval["neighbors"]):
            raise ValueError("retrieval.neighbors must be a positive integer.")
        if type(retrieval["rrf_k"]) is not int or retrieval["rrf_k"] < 0:
            raise ValueError("retrieval.rrf_k must be a nonnegative integer.")
        for key in ("sparse", "dense"):
            if type(retrieval.get(key, True)) is not bool:
                raise ValueError(f"retrieval.{key} must be a boolean.")
        if not (retrieval.get("sparse", True) or retrieval.get("dense", True)):
            raise ValueError("Enable at least one retriever (sparse or dense).")
        if not retrieval.get("dense", True):
            return
        encoder = self.config.get("encoder")
        revision = encoder.get("revision") if isinstance(encoder, dict) else None
        if not (isinstance(revision, str) and COMMIT_SHA.fullmatch(revision)):
            raise ValueError(
                "Dense retrieval needs encoder.revision pinned to a full 40-character "
                f"lowercase commit SHA, not {revision!r}."
            )

    def _fit(
        self, features: spmatrix, targets: np.ndarray, texts: list[str], groups: Sequence[str]
    ) -> dict:
        weights: dict[str, Any] = {"targets": targets.astype(np.float32)}
        if self.config["retrieval"].get("sparse", True):
            weights["sparse"] = features.tocsr()
        if self.config["retrieval"].get("dense", True):
            settings = self.config["encoder"]
            weights["dense"] = self.encoder(texts).astype(np.float16)
            weights["fingerprint"] = {
                "pretrained_name": settings["pretrained_name"],
                "revision": settings["revision"],
                "canary": self.encoder(list(CANARY_TEXTS)).astype(np.float32),
            }
        return weights

    def _check_weights(self, weights: dict, label_count: int) -> None:
        if weights["targets"].shape[1] != label_count:
            raise ValueError("Neighbour index and artifact label space disagree.")
        # Scoring uses whichever indexes the weights hold, so they must be the configured ones.
        retrieval = self.config["retrieval"]
        enabled = {key for key in ("sparse", "dense") if retrieval.get(key, True)}
        if enabled != weights.keys() & {"sparse", "dense"}:
            raise ValueError("Artifact index and retrieval config disagree; retrain.")
        if "dense" not in weights:
            return
        fingerprint = weights.get("fingerprint")
        if (
            not isinstance(fingerprint, dict)
            or {"pretrained_name", "revision", "canary"} - set(fingerprint)
            or not isinstance(fingerprint["canary"], np.ndarray)
        ):
            raise ValueError("Dense artifact lacks an encoder fingerprint; retrain the model.")
        settings = self.config["encoder"]
        configured = settings["pretrained_name"], settings.get("revision")
        fingerprinted = fingerprint["pretrained_name"], fingerprint["revision"]
        if configured != fingerprinted:
            raise ValueError(
                "Artifact config encoder {}@{} differs from its fingerprint {}@{}; "
                "retrain the model.".format(*configured, *fingerprinted)
            )

    def _verify_encoder(self) -> None:
        """Refuse a query encoder that does not reproduce the artifact's canary embeddings."""
        fingerprint = self.weights["fingerprint"]
        canary = self.encoder(list(CANARY_TEXTS)).astype(np.float32)
        if canary.shape != fingerprint["canary"].shape or not np.allclose(
            canary, fingerprint["canary"], atol=CANARY_ATOL
        ):
            raise ValueError(
                f"Query encoder {fingerprint['pretrained_name']}@{fingerprint['revision']} does "
                "not reproduce the artifact's canary embeddings, so it cannot search the saved "
                "document embeddings; load that exact encoder or retrain."
            )
        self._encoder_verified = True

    def _score_matrix(self, features: spmatrix, texts: Sequence[str]) -> np.ndarray:
        """Fused neighbour label votes; a query with no candidate abstains with all-zero scores."""
        retrieval = self.config["retrieval"]
        targets = self.weights["targets"]
        neighbors = min(retrieval["neighbors"], len(targets))
        if "dense" in self.weights and not self._encoder_verified:
            self._verify_encoder()
        scores = []
        for start in range(0, len(texts), QUERY_CHUNK):
            stop = start + QUERY_CHUNK
            similarities = []
            if "sparse" in self.weights:
                similarities.append((features[start:stop] @ self.weights["sparse"].T).toarray())
            if "dense" in self.weights:
                query = self.encoder(list(texts[start:stop])).astype(np.float32)
                similarities.append(query @ self.weights["dense"].astype(np.float32).T)
            fused = reciprocal_rank_fusion(similarities, neighbors, retrieval["rrf_k"])
            votes = np.maximum(fused.sum(axis=1, keepdims=True), np.finfo(np.float64).tiny)
            scores.append(fused @ targets / votes)
        return np.concatenate(scores)
