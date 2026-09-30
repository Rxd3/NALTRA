"""Per-label voting over schema-compatible NALTRA model predictions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from naltra.pipeline.thresholds import apply_thresholds
from naltra.schemas.prediction import Explanation, LabelScore, PredictionResult

ENSEMBLE_MODEL_NAMES = frozenset({"naive_bayes", "svm", "bilstm", "transformer", "jev", "laya"})


class EnsembleMethod(StrEnum):
    """Supported methods for combining per-label model decisions."""

    HARD = "hard"
    SOFT = "soft"
    WEIGHTED_SOFT = "weighted_soft"


@dataclass(frozen=True, slots=True)
class EnsembleConfig:
    """Validated runtime configuration for ensemble voting."""

    method: EnsembleMethod | str = EnsembleMethod.WEIGHTED_SOFT
    threshold: float = 0.5
    weights: Mapping[str, float] = field(default_factory=dict)
    per_label_thresholds: Mapping[str, float] = field(default_factory=dict)
    required_models: frozenset[str] | Iterable[str] = ENSEMBLE_MODEL_NAMES

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", EnsembleMethod(self.method))
        object.__setattr__(self, "required_models", frozenset(self.required_models))
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Ensemble threshold must be between 0.0 and 1.0.")
        invalid_label_thresholds = {
            label: threshold
            for label, threshold in self.per_label_thresholds.items()
            if not 0.0 <= threshold <= 1.0
        }
        if invalid_label_thresholds:
            raise ValueError("Per-label thresholds must be between 0.0 and 1.0.")
        if not self.required_models:
            raise ValueError("At least one required ensemble model must be configured.")


class EnsembleVoter:
    """Combine one prediction per configured model into one PredictionResult.

    Voting is performed independently for every label. A missing label is treated
    as probability zero for soft voting and as a negative vote for hard voting.
    """

    def __init__(self, config: EnsembleConfig) -> None:
        self.config = config

    def combine(self, predictions: Sequence[PredictionResult]) -> PredictionResult:
        """Validate and combine predictions from all configured model families."""
        predictions_by_model = self._validate_predictions(predictions)
        score_maps = {
            model: {label_score.label: label_score.score for label_score in prediction.labels}
            for model, prediction in predictions_by_model.items()
        }
        labels = sorted(set().union(*(scores.keys() for scores in score_maps.values())))

        if self.config.method is EnsembleMethod.HARD:
            scores = self._hard_vote(labels, score_maps)
            selected = self._select_hard_majorities(scores)
        elif self.config.method is EnsembleMethod.SOFT:
            scores = self._soft_vote(labels, score_maps)
            selected = apply_thresholds(
                scores,
                default_threshold=self.config.threshold,
                per_label=self.config.per_label_thresholds,
            )
        else:
            weights = self._validated_weights(predictions_by_model)
            scores = self._weighted_soft_vote(labels, score_maps, weights)
            selected = apply_thresholds(
                scores,
                default_threshold=self.config.threshold,
                per_label=self.config.per_label_thresholds,
            )

        first_prediction = predictions[0]
        return PredictionResult(
            text=first_prediction.text,
            model="ensemble",
            language=first_prediction.language,
            labels=selected,
            latency_ms=sum(prediction.latency_ms for prediction in predictions),
            explanation=Explanation(
                important_tokens=self._merge_important_tokens(predictions),
                details={
                    "method": self.config.method.value,
                    "models": sorted(predictions_by_model),
                    "weights": dict(self.config.weights),
                    "label_scores": scores,
                },
            ),
        )

    def _validate_predictions(
        self, predictions: Sequence[PredictionResult]
    ) -> dict[str, PredictionResult]:
        if not predictions:
            raise ValueError("Ensemble voting requires model predictions.")

        predictions_by_model = {prediction.model: prediction for prediction in predictions}
        if len(predictions_by_model) != len(predictions):
            raise ValueError("Ensemble predictions must contain unique model names.")

        supplied_models = set(predictions_by_model)
        missing = self.config.required_models - supplied_models
        unexpected = supplied_models - self.config.required_models
        if missing or unexpected:
            parts: list[str] = []
            if missing:
                parts.append("missing: " + ", ".join(sorted(missing)))
            if unexpected:
                parts.append("unexpected: " + ", ".join(sorted(unexpected)))
            raise ValueError("Ensemble model set mismatch (" + "; ".join(parts) + ").")

        reference = predictions[0]
        if any(prediction.text != reference.text for prediction in predictions[1:]):
            raise ValueError("All ensemble predictions must refer to the same text.")
        if any(prediction.language != reference.language for prediction in predictions[1:]):
            raise ValueError("All ensemble predictions must use the same language metadata.")
        return predictions_by_model

    def _validated_weights(
        self, predictions_by_model: Mapping[str, PredictionResult]
    ) -> dict[str, float]:
        supplied = set(self.config.weights)
        expected = set(predictions_by_model)
        if supplied != expected:
            missing = expected - supplied
            unexpected = supplied - expected
            parts: list[str] = []
            if missing:
                parts.append("missing: " + ", ".join(sorted(missing)))
            if unexpected:
                parts.append("unexpected: " + ", ".join(sorted(unexpected)))
            raise ValueError("Ensemble weight set mismatch (" + "; ".join(parts) + ").")

        weights = dict(self.config.weights)
        if any(weight < 0.0 for weight in weights.values()):
            raise ValueError("Ensemble weights cannot be negative.")
        if sum(weights.values()) <= 0.0:
            raise ValueError("At least one ensemble weight must be positive.")
        return weights

    @staticmethod
    def _hard_vote(
        labels: Sequence[str], score_maps: Mapping[str, Mapping[str, float]]
    ) -> dict[str, float]:
        model_count = len(score_maps)
        return {
            label: sum(label in scores for scores in score_maps.values()) / model_count
            for label in labels
        }

    @staticmethod
    def _soft_vote(
        labels: Sequence[str], score_maps: Mapping[str, Mapping[str, float]]
    ) -> dict[str, float]:
        model_count = len(score_maps)
        return {
            label: sum(scores.get(label, 0.0) for scores in score_maps.values()) / model_count
            for label in labels
        }

    @staticmethod
    def _weighted_soft_vote(
        labels: Sequence[str],
        score_maps: Mapping[str, Mapping[str, float]],
        weights: Mapping[str, float],
    ) -> dict[str, float]:
        total_weight = sum(weights.values())
        return {
            label: sum(score_maps[model].get(label, 0.0) * weights[model] for model in score_maps)
            / total_weight
            for label in labels
        }

    def _select_hard_majorities(self, scores: Mapping[str, float]) -> list[LabelScore]:
        selected = [
            LabelScore(label=label, score=score)
            for label, score in scores.items()
            if score > 0.5
            and score >= self.config.per_label_thresholds.get(label, self.config.threshold)
        ]
        return sorted(selected, key=lambda item: item.score, reverse=True)

    @staticmethod
    def _merge_important_tokens(predictions: Sequence[PredictionResult]) -> list[str]:
        tokens: list[str] = []
        for prediction in predictions:
            if prediction.explanation is None:
                continue
            for token in prediction.explanation.important_tokens:
                if token not in tokens:
                    tokens.append(token)
        return tokens
