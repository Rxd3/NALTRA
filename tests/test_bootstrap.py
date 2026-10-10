"""Paired project-level bootstrap for F1 differences, with Holm correction."""

from __future__ import annotations

import numpy as np
import pytest

from naltra.evaluation.bootstrap import f1_scores, holm, paired_bootstrap, resample_counts
from naltra.evaluation.matrix import flat_metrics


def test_resample_counts_draw_n_documents_per_replicate() -> None:
    counts = resample_counts(10, resamples=50, seed=0)
    assert counts.shape == (50, 10)
    assert (counts.sum(axis=1) == 10).all()
    assert np.array_equal(counts, resample_counts(10, resamples=50, seed=0))


def test_weighted_f1_matches_flat_metrics_on_unit_weights() -> None:
    rng = np.random.default_rng(1)
    truth = rng.random((40, 6)) < 0.3
    predicted = rng.random((40, 6)) < 0.3
    micro, macro = f1_scores(np.ones((1, 40)), truth, predicted)
    reference = flat_metrics(truth, predicted)
    assert micro[0] == pytest.approx(reference["micro_f1"])
    assert macro[0] == pytest.approx(reference["macro_f1"])


def test_clearly_better_system_has_positive_interval_and_small_p() -> None:
    rng = np.random.default_rng(2)
    truth = rng.random((300, 8)) < 0.3
    good = truth ^ (rng.random(truth.shape) < 0.05)
    bad = truth ^ (rng.random(truth.shape) < 0.30)
    result = paired_bootstrap(truth, good, truth, bad, resamples=500, seed=3)
    for metric in ("micro_f1", "macro_f1"):
        assert result[metric]["difference"] > 0
        assert result[metric]["ci_low"] > 0
        assert result[metric]["p_value"] < 0.01


def test_identical_systems_are_not_significant() -> None:
    rng = np.random.default_rng(4)
    truth = rng.random((200, 5)) < 0.3
    predicted = truth ^ (rng.random(truth.shape) < 0.2)
    result = paired_bootstrap(truth, predicted, truth, predicted, resamples=200, seed=5)
    assert result["micro_f1"]["difference"] == 0
    assert result["micro_f1"]["p_value"] == pytest.approx(1.0)


def test_rows_must_be_paired() -> None:
    truth = np.zeros((3, 2), dtype=bool)
    with pytest.raises(ValueError, match="paired"):
        paired_bootstrap(truth, truth, truth[:2], truth[:2])


def projects(correct: range, count: int) -> tuple[np.ndarray, np.ndarray]:
    truth = np.tile([True, False], (count, 1))
    return truth, np.array([[p in correct, p not in correct] for p in range(count)])


def test_duplicated_variants_of_a_project_resample_as_one_project() -> None:
    truth, a = projects(range(15), 20)
    _, b = projects(range(12, 20), 20)
    single = paired_bootstrap(truth, a, truth, b, resamples=1000, seed=42)
    twice = np.repeat(np.arange(20), 2)
    doubled = paired_bootstrap(
        truth[twice], a[twice], truth[twice], b[twice], resamples=1000, seed=42, groups=twice
    )
    assert single["micro_f1"]["ci_low"] < 0
    for metric in ("micro_f1", "macro_f1"):
        assert doubled[metric] == pytest.approx(single[metric])


def test_groups_must_align_with_rows() -> None:
    truth = np.zeros((3, 2), dtype=bool)
    with pytest.raises(ValueError, match="group"):
        paired_bootstrap(truth, truth, truth, truth, groups=[0, 1])


def test_unobserved_tail_is_bounded_by_resampling_resolution() -> None:
    truth, a = projects(range(10), 11)
    _, b = projects(range(10, 11), 11)
    result = paired_bootstrap(truth, a, truth, b, resamples=1000, seed=42)
    for metric in ("micro_f1", "macro_f1"):
        assert result[metric]["p_resolution"] == pytest.approx(2 / 1001)
        assert result[metric]["p_value"] >= result[metric]["p_resolution"]
    assert result["micro_f1"]["p_value"] == pytest.approx(2 / 1001)


def test_holm_adjusts_step_down_and_stays_monotone() -> None:
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["c"] == pytest.approx(0.06)
    assert adjusted["b"] == pytest.approx(0.06)
