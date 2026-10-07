from dataclasses import replace

import pytest
from tests.test_pipeline import EchoModel

from naltra.pipeline.ensemble import EnsembleConfig, EnsembleVoter
from naltra.pipeline.language import LinguaLanguageDetector, UndeterminedLanguageDetector
from naltra.pipeline.ood import MaxProbabilityOODDetector
from naltra.pipeline.prediction import PredictionPipeline
from naltra.pipeline.thresholds import apply_thresholds
from naltra.schemas.prediction import Explanation, LabelScore, LanguageInfo, PredictionResult


class ScoredModel(EchoModel):
    def __init__(self, name="bilstm"):
        self.name = name

    def predict(self, text):
        return PredictionResult(
            text=text,
            model=self.name,
            language=LanguageInfo("und"),
            labels=[LabelScore("child", 0.8)],
            label_scores={"child": 0.8, "root": 0.2},
            metadata={"ood_threshold": 0.1},
        )


def test_pipeline_thresholds_paths_ood_and_hooks():
    pipeline = PredictionPipeline(
        ScoredModel(),
        parent_by_label={"root": None, "child": "root"},
        language_detector=UndeterminedLanguageDetector(),
        threshold=0.5,
        per_label_thresholds={"root": 0.1},
        calibrator=lambda scores: dict(scores),
        ood_threshold=0.1,
        explainer=lambda result: Explanation(["token"]),
    )
    result = pipeline.predict("  some   text ", language=LanguageInfo("tr"))
    assert result.language.primary == "tr"
    assert result.hierarchy_paths == [["root"], ["root", "child"]]
    assert result.ood.is_ood
    assert result.ood.score == pytest.approx(0.2)
    assert result.metadata["calibration"] == "custom"
    assert result.explanation.important_tokens == ["token"]
    assert result.latency_ms > 0


def test_soft_ensemble_uses_below_threshold_probabilities():
    def prediction(name):
        return PredictionResult(
            text="text",
            model=name,
            language=LanguageInfo("en"),
            label_scores={"sport": 0.4},
            labels=[],
        )

    result = EnsembleVoter(
        EnsembleConfig(method="soft", threshold=0.3, required_models={"bilstm", "transformer"})
    ).combine([prediction("bilstm"), prediction("transformer")])
    assert result.label_scores == {"sport": 0.4}
    assert result.labels == [LabelScore("sport", 0.4)]
    hard = EnsembleVoter(
        EnsembleConfig(method="hard", required_models={"bilstm", "transformer"})
    ).combine([prediction("bilstm"), prediction("transformer")])
    assert hard.labels == []


def test_ensemble_pipeline_and_missing_components():
    config = EnsembleConfig(method="soft", required_models={"bilstm", "transformer"})
    with pytest.raises(ValueError, match="missing"):
        PredictionPipeline(models={"bilstm": ScoredModel()}, ensemble_config=config)
    pipeline = PredictionPipeline(
        models={name: ScoredModel(name) for name in config.required_models},
        ensemble_config=config,
        parent_by_label={"root": None, "child": "root"},
        language_detector=UndeterminedLanguageDetector(),
        ood_threshold=0.1,
    )
    result = pipeline.predict("text")
    assert result.model == "ensemble"
    assert result.hierarchy_paths == [["root", "child"]]
    assert result.ood.is_ood


def test_ensemble_rejects_different_experiments():
    first = replace(ScoredModel().predict("text"), metadata={"track": "clean"})
    second = replace(ScoredModel("transformer").predict("text"), metadata={"track": "heldout"})
    voter = EnsembleVoter(EnsembleConfig(method="soft", required_models={"bilstm", "transformer"}))
    with pytest.raises(ValueError, match="track mismatch"):
        voter.combine([first, second])


def test_pipeline_batch_validation_and_unknown_labels():
    pipeline = PredictionPipeline(EchoModel(), language_detector=UndeterminedLanguageDetector())
    assert pipeline.predict_batch([]) == []
    with pytest.raises(ValueError, match="batch length"):
        pipeline.predict_batch(["one", "two"], languages=[LanguageInfo("en")])
    with pytest.raises(ValueError, match="empty"):
        pipeline.predict_batch(["valid", " "])
    with pytest.raises(ValueError, match="Unknown canonical"):
        PredictionPipeline(ScoredModel(), language_detector=UndeterminedLanguageDetector()).predict(
            "text"
        )


@pytest.mark.parametrize(
    "text, language, switched",
    [
        ("Football players compete in the championship tournament", "en", False),
        ("Türkiye ekonomisinde gelişmeler ve araştırmalar devam ediyor", "tr", False),
        (
            "Football players compete in the championship tournament. "
            "Türkiye ekonomisinde gelişmeler ve araştırmalar devam ediyor.",
            "en-tr",
            True,
        ),
        ("123 456 !!!", "und", False),
        ("hello", "und", False),
    ],
)
def test_offline_language_detection(text, language, switched):
    result = LinguaLanguageDetector().detect(text)
    assert result.primary == language
    assert result.is_code_switched == switched


def test_score_and_threshold_validation():
    for scores, default, per_label in [
        ({"sport": float("nan")}, 0.5, {}),
        ({"sport": -0.1}, 0.5, {}),
        ({}, 0.5, {"sport": float("inf")}),
    ]:
        with pytest.raises(ValueError):
            apply_thresholds(scores, default, per_label)
    with pytest.raises(ValueError, match="unique"):
        PredictionResult("text", "model", LanguageInfo("en"), labels=[LabelScore("sport", 0.5)] * 2)
    with pytest.raises(ValueError, match="finite"):
        PredictionResult("text", "model", LanguageInfo("en"), latency_ms=float("nan"))
    detector = MaxProbabilityOODDetector(0.2)
    assert not detector.detect_scores({"sport": 0.8}).is_ood
    assert detector.detect_scores({"sport": 0.7}).is_ood
    with pytest.raises(ValueError, match="complete"):
        detector.detect_scores({})


def test_calibration_disables_unfitted_ood_threshold():
    pipeline = PredictionPipeline(
        ScoredModel(),
        parent_by_label={"root": None, "child": "root"},
        language_detector=UndeterminedLanguageDetector(),
        calibrator=lambda scores: dict(scores),
    )
    result = pipeline.predict("some text")
    assert result.metadata["ood_method"] == "disabled"
    assert result.metadata["ood_threshold"] is None


def test_hard_pipeline_thresholds_only_filter_majorities():
    config = EnsembleConfig(method="hard", required_models={"bilstm", "transformer"})
    pipeline = PredictionPipeline(
        models={name: ScoredModel(name) for name in config.required_models},
        ensemble_config=config,
        parent_by_label={"root": None, "child": "root"},
        language_detector=UndeterminedLanguageDetector(),
        per_label_thresholds={"root": 0},
    )
    result = pipeline.predict("some text")
    assert [item.label for item in result.labels] == ["child"]
    assert result.metadata["score_semantics"] == "vote_fraction"
    with pytest.raises(ValueError, match="custom OOD"):
        PredictionPipeline(models=pipeline.models, ensemble_config=config, ood_threshold=0.5)


def test_ensemble_rejects_registry_name_mismatch():
    config = EnsembleConfig(method="soft", required_models={"bilstm", "transformer"})
    pipeline = PredictionPipeline(
        models={"bilstm": ScoredModel(), "transformer": ScoredModel()},
        ensemble_config=config,
        language_detector=UndeterminedLanguageDetector(),
    )
    with pytest.raises(ValueError, match="model name"):
        pipeline.predict("some text")
