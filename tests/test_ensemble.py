from dataclasses import replace

import pytest

from naltra.data.manifest import REPO_ROOT
from naltra.pipeline.ensemble import (
    ENSEMBLE_MODEL_NAMES,
    EnsembleConfig,
    EnsembleMethod,
    EnsembleVoter,
)
from naltra.pipeline.hierarchy import resolve_hierarchy_paths
from naltra.schemas.prediction import LabelScore, LanguageInfo, OODResult, PredictionResult
from naltra.utils.config import load_yaml

MODEL_NAMES = ["naive_bayes", "svm", "bilstm", "transformer", "kev", "jev", "laya", "hybrid_knn"]


def make_prediction(model: str, scores: dict[str, float]) -> PredictionResult:
    return PredictionResult(
        text="A multilingual chip announcement",
        model=model,
        language=LanguageInfo(primary="en", is_code_switched=False),
        labels=[LabelScore(label=label, score=score) for label, score in scores.items()],
        latency_ms=1.0,
    )


def test_hard_voting_requires_a_strict_per_label_majority() -> None:
    predictions = [
        make_prediction("naive_bayes", {"Technology": 0.7, "Hardware": 0.6}),
        make_prediction("svm", {"Technology": 0.8, "Hardware": 0.7}),
        make_prediction("bilstm", {"Technology": 0.9, "Hardware": 0.8}),
        make_prediction("transformer", {"Technology": 0.9}),
        make_prediction("kev", {}),
        make_prediction("jev", {}),
        make_prediction("laya", {"Technology": 0.9}),
        make_prediction("hybrid_knn", {}),
    ]

    result = EnsembleVoter(EnsembleConfig(method="hard")).combine(predictions)

    assert result.model == "ensemble"
    assert [label.label for label in result.labels] == ["Technology"]
    assert result.labels[0].score == pytest.approx(5 / 8)
    assert result.latency_ms == pytest.approx(8.0)


def test_soft_voting_averages_each_label_and_supports_per_label_thresholds() -> None:
    predictions = [
        make_prediction(model, {"Technology": 0.6, "Hardware": 0.4}) for model in MODEL_NAMES
    ]
    config = EnsembleConfig(
        method=EnsembleMethod.SOFT,
        threshold=0.5,
        per_label_thresholds={"Hardware": 0.4},
    )

    result = EnsembleVoter(config).combine(predictions)

    assert {label.label: label.score for label in result.labels} == {
        "Technology": pytest.approx(0.6),
        "Hardware": pytest.approx(0.4),
    }


def test_weighted_soft_voting_uses_configured_model_weights() -> None:
    predictions = [
        make_prediction(model, {"Technology": 1.0} if model == "hybrid_knn" else {})
        for model in MODEL_NAMES
    ]
    weights = {model: 1.0 for model in MODEL_NAMES}
    weights["hybrid_knn"] = 7.0
    config = EnsembleConfig(
        method="weighted_soft",
        threshold=0.4,
        weights=weights,
    )

    result = EnsembleVoter(config).combine(predictions)

    assert result.labels == [LabelScore(label="Technology", score=0.5)]
    assert result.explanation is not None
    assert result.explanation.details["weights"]["hybrid_knn"] == 7.0


def test_ensemble_result_can_continue_through_hierarchy_and_ood_processing() -> None:
    predictions = [make_prediction(model, {"Semiconductors": 0.9}) for model in MODEL_NAMES]
    result = EnsembleVoter(EnsembleConfig(method="soft")).combine(predictions)
    parents = {
        "Technology": None,
        "Hardware": "Technology",
        "Semiconductors": "Hardware",
    }

    processed = replace(
        result,
        hierarchy_paths=resolve_hierarchy_paths([label.label for label in result.labels], parents),
        ood=OODResult(is_ood=True, score=0.8),
    )

    assert processed.hierarchy_paths == [["Technology", "Hardware", "Semiconductors"]]
    assert processed.ood.is_ood


def test_ensemble_requires_every_configured_model() -> None:
    predictions = [make_prediction(model, {}) for model in MODEL_NAMES[:-1]]

    with pytest.raises(ValueError, match="missing: hybrid_knn"):
        EnsembleVoter(EnsembleConfig(method="soft")).combine(predictions)


def test_weighted_voting_requires_one_weight_per_model() -> None:
    predictions = [make_prediction(model, {}) for model in MODEL_NAMES]

    with pytest.raises(ValueError, match="weight set mismatch"):
        EnsembleVoter(EnsembleConfig(method="weighted_soft", weights={"hybrid_knn": 1.0})).combine(
            predictions
        )


def test_default_ensemble_model_set_matches_the_benchmark_members() -> None:
    systems = load_yaml(REPO_ROOT / "configs/benchmark.yaml")["systems"]
    members = {
        name for name, entry in systems.items() if not entry["architecture"].startswith("NALTRA")
    }
    assert ENSEMBLE_MODEL_NAMES == members == set(MODEL_NAMES)
