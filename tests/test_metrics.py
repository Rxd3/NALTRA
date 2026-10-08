import pytest

from naltra.evaluation.calibration import brier_score
from naltra.evaluation.metrics import (
    compute_classification_metrics,
    multilabel_metrics,
    precision_recall_f1,
)
from naltra.evaluation.robustness import jaccard_stability
from naltra.schemas.prediction import LabelScore, LanguageInfo, PredictionResult


def test_set_precision_recall_f1() -> None:
    precision, recall, f1 = precision_recall_f1({"a", "b"}, {"a", "c"})
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert f1 == pytest.approx(0.5)


def test_multilabel_micro_and_macro_metrics() -> None:
    metrics = multilabel_metrics([{"a", "b"}, {"a"}], [{"a"}, {"a", "c"}])
    assert metrics["micro_precision"] == pytest.approx(2 / 3)
    assert metrics["micro_recall"] == pytest.approx(2 / 3)
    assert metrics["micro_f1"] == pytest.approx(2 / 3)
    assert metrics["macro_f1"] == pytest.approx(1 / 3)


def test_reliability_helpers_are_lightweight() -> None:
    assert brier_score([0.0, 1.0], [0, 1]) == 0.0
    assert jaccard_stability({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


def test_macro_uses_fixed_label_universe_including_absent_labels():
    metrics = multilabel_metrics([{"a"}], [{"a"}], all_labels=["a", "b"])
    assert metrics["micro_f1"] == 1.0
    assert metrics["macro_f1"] == 0.5
    with pytest.raises(ValueError, match="universe"):
        multilabel_metrics([{"unknown"}], [set()], all_labels=["a", "b"])


def test_lower_threshold_recovers_labels_from_full_scores():
    prediction = PredictionResult(
        text="one",
        model="fixture",
        language=LanguageInfo("und"),
        labels=[LabelScore("a", 0.8)],
        label_scores={"a": 0.8, "b": 0.4},
    )
    result = compute_classification_metrics([prediction], [{"a", "b"}], ["a", "b"], 0.3)
    assert result["micro_f1"] == result["macro_f1"] == 1.0


def test_multilabel_metrics_match_sklearn_on_fixed_universe():
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support

    random = np.random.default_rng(42)
    truth = random.random((30, 7)) > 0.8
    predicted = random.random((30, 7)) > 0.7
    truth[:, -1] = predicted[:, -1] = False
    labels = [str(i) for i in range(7)]
    result = multilabel_metrics(
        [{label for label, present in zip(labels, row, strict=True) if present} for row in truth],
        [
            {label for label, present in zip(labels, row, strict=True) if present}
            for row in predicted
        ],
        labels,
    )
    for average in ("micro", "macro"):
        expected = precision_recall_fscore_support(
            truth, predicted, average=average, zero_division=0
        )
        for metric, value in zip(("precision", "recall", "f1"), expected[:3], strict=True):
            assert result[f"{average}_{metric}"] == pytest.approx(value)
