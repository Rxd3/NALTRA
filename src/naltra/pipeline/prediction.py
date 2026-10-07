"""Shared model/ensemble decisions and postprocessing."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter

from naltra.models.base import BaseNALTRAModel
from naltra.pipeline.ensemble import EnsembleConfig, EnsembleMethod, EnsembleVoter
from naltra.pipeline.hierarchy import resolve_hierarchy_paths, validate_hierarchy
from naltra.pipeline.language import LanguageDetector, LinguaLanguageDetector
from naltra.pipeline.ood import MaxProbabilityOODDetector, OODDetector
from naltra.pipeline.preprocessing import normalize_text
from naltra.pipeline.thresholds import apply_thresholds
from naltra.schemas.prediction import Explanation, LanguageInfo, OODResult, PredictionResult


@dataclass(slots=True)
class PredictionPipeline:
    """Prepare text and delegate prediction to the selected model."""

    model: BaseNALTRAModel | None = None
    preprocessor: Callable[[str], str] = normalize_text
    models: Mapping[str, BaseNALTRAModel] = field(default_factory=dict)
    ensemble_config: EnsembleConfig | None = None
    language_detector: LanguageDetector = field(default_factory=LinguaLanguageDetector)
    parent_by_label: Mapping[str, str | None] | None = None
    threshold: float | None = None
    per_label_thresholds: Mapping[str, float] = field(default_factory=dict)
    ood_threshold: float | None = None
    ood_detector: OODDetector | None = None
    calibrator: Callable[[Mapping[str, float]], Mapping[str, float]] | None = None
    explainer: Callable[[PredictionResult], Explanation] | None = None

    def __post_init__(self) -> None:
        if (self.model is None) == (self.ensemble_config is None):
            raise ValueError("Configure exactly one model or an ensemble.")
        if self.ensemble_config is not None:
            supplied = set(self.models)
            required = set(self.ensemble_config.required_models)
            if supplied != required:
                raise ValueError(
                    f"Unavailable ensemble components: missing {sorted(required - supplied)}, "
                    f"unexpected {sorted(supplied - required)}."
                )
        if self.parent_by_label is None:
            path = Path(__file__).resolve().parents[3] / "taxonomy/taxonomy.json"
            taxonomy = json.loads(path.read_text(encoding="utf-8"))
            self.parent_by_label = {item["id"]: item["parent"] for item in taxonomy["labels"]}
        errors = validate_hierarchy(self.parent_by_label)
        if errors:
            raise ValueError("Invalid hierarchy: " + "; ".join(errors))
        apply_thresholds(
            {}, 0.5 if self.threshold is None else self.threshold, self.per_label_thresholds
        )
        if self.ood_threshold is not None:
            MaxProbabilityOODDetector(self.ood_threshold)
        if self.ensemble_config and self.ensemble_config.method is EnsembleMethod.HARD:
            if self.calibrator:
                raise ValueError("Hard-vote fractions cannot use probability calibration.")
            if self.ood_threshold is not None and self.ood_detector is None:
                raise ValueError(
                    "Hard voting needs a custom OOD detector, not a probability threshold."
                )

    def predict(self, text: str, *, language: LanguageInfo | None = None) -> PredictionResult:
        return self.predict_batch([text], languages=[language] if language else None)[0]

    def predict_batch(
        self, texts: Iterable[str], *, languages: Iterable[LanguageInfo] | None = None
    ) -> list[PredictionResult]:
        started = perf_counter()
        prepared = [self.preprocessor(text) for text in texts]
        if any(not text for text in prepared):
            raise ValueError("Prediction texts cannot contain empty values.")
        known = list(languages) if languages is not None else None
        if known is not None and len(known) != len(prepared):
            raise ValueError("Language overrides must match batch length.")
        if not prepared:
            return []
        detected = (
            known
            if known is not None
            else [self.language_detector.detect(text) for text in prepared]
        )
        if self.ensemble_config is not None:
            batches = {name: model.predict_batch(prepared) for name, model in self.models.items()}
            if any(len(batch) != len(prepared) for batch in batches.values()):
                raise ValueError("Model returned an incorrect batch length.")
            for name, batch in batches.items():
                if any(result.model != name for result in batch):
                    raise ValueError(
                        f"Ensemble component {name!r} returned a different model name."
                    )
            voter = EnsembleVoter(self.ensemble_config)
            results = [
                voter.combine(
                    [replace(batch[index], language=detected[index]) for batch in batches.values()]
                )
                for index in range(len(prepared))
            ]
        else:
            assert self.model is not None
            results = self.model.predict_batch(prepared)
        if len(results) != len(prepared):
            raise ValueError("Model returned an incorrect batch length.")
        processed = [
            self._finish(result, text, language)
            for result, text, language in zip(results, prepared, detected, strict=True)
        ]
        elapsed = (perf_counter() - started) * 1000 / len(prepared)
        return [replace(result, latency_ms=elapsed) for result in processed]

    def _finish(
        self, result: PredictionResult, text: str, language: LanguageInfo
    ) -> PredictionResult:
        if result.text != text:
            raise ValueError("Model returned predictions out of order or for different text.")
        scores = dict(result.label_scores)
        labels = result.labels
        metadata = dict(result.metadata)
        if self.calibrator is not None:
            if not scores:
                raise ValueError("Calibration requires complete label probabilities.")
            calibrated = dict(self.calibrator(scores))
            if set(calibrated) != set(scores):
                raise ValueError("Calibration must preserve label space.")
            scores = calibrated
            metadata["calibration"] = "custom"
        hard = self.ensemble_config and self.ensemble_config.method is EnsembleMethod.HARD
        if (
            not hard
            and scores
            and (self.threshold is not None or self.per_label_thresholds or self.calibrator)
        ):
            if self.threshold is not None:
                threshold = self.threshold
            elif self.ensemble_config:
                threshold = self.ensemble_config.threshold
            else:
                threshold = (
                    getattr(self.model, "config", {}).get("multilabel", {}).get("threshold", 0.5)
                )
            labels = apply_thresholds(scores, threshold, self.per_label_thresholds)
        elif hard and (self.threshold is not None or self.per_label_thresholds):
            # Only filter strict majorities; never promote ties or negative votes.
            labels = apply_thresholds(
                {item.label: item.score for item in labels},
                self.threshold if self.threshold is not None else self.ensemble_config.threshold,
                self.per_label_thresholds,
            )
        # Check even unselected probabilities against the canonical taxonomy.
        assert self.parent_by_label is not None
        unknown = (set(scores) | {label.label for label in labels}) - set(self.parent_by_label)
        if unknown:
            raise ValueError(f"Unknown canonical labels: {sorted(unknown)}.")
        paths = resolve_hierarchy_paths([label.label for label in labels], self.parent_by_label)
        ood = OODResult(is_ood=False, score=0.0)
        threshold = (
            self.ood_threshold if self.ood_threshold is not None else metadata.get("ood_threshold")
        )
        if self.calibrator is not None and self.ood_threshold is None:
            # The model's threshold was fitted to uncalibrated scores.
            threshold = None
        metadata["ood_threshold"] = threshold
        if self.ood_detector is not None:
            ood = self.ood_detector.detect(text)
            metadata["ood_method"] = "custom"
        elif threshold is not None and scores:
            if hard:
                raise ValueError(
                    "Hard-vote fractions are not probabilities; configure a custom OOD detector."
                )
            ood = MaxProbabilityOODDetector(threshold).detect_scores(scores)
            metadata["ood_method"] = "max_probability"
        else:
            metadata["ood_method"] = "disabled"
        result = replace(
            result,
            language=language,
            labels=labels,
            label_scores=scores,
            hierarchy_paths=paths,
            ood=ood,
            metadata=metadata,
        )
        if self.explainer is not None:
            result = replace(result, explanation=self.explainer(result))
        return result
