import pytest

from naltra.evaluation.calibration import brier_score
from naltra.evaluation.metrics import multilabel_metrics, precision_recall_f1
from naltra.evaluation.robustness import jaccard_stability


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
