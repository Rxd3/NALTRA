"""Numbers behind the paper figures: root confusion, F1 by depth, reliability bins."""

from __future__ import annotations

import numpy as np
import pytest

from naltra.evaluation.figures import depth_f1, reliability_bins, root_confusion

NODES = ["r1", "r2", "a", "b"]
PARENTS = {"r1": None, "r2": None, "a": "r1", "b": "r2"}


def test_root_confusion_rows_are_gold_roots_and_columns_predicted_roots() -> None:
    truth = np.array([[1, 0, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1]])
    predicted = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 1]])
    roots, matrix = root_confusion(truth, predicted, NODES, PARENTS)
    assert roots == ["r1", "r2"]
    # Gold r1 (2 docs): predicted r1 once, r2 once. Gold r2 (1 doc): predicted both.
    assert matrix.tolist() == [[0.5, 0.5], [1.0, 1.0]]


def test_root_confusion_does_not_count_a_correct_second_root_as_an_error() -> None:
    both = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])
    _, matrix = root_confusion(both, both, NODES, PARENTS)
    assert matrix.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_root_confusion_off_diagonal_is_the_false_root_rate() -> None:
    truth = np.array([[1, 0, 1, 0], [1, 1, 1, 1]])
    predicted = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])
    _, matrix = root_confusion(truth, predicted, NODES, PARENTS)
    # Gold r1 (2 docs): r2 is a false root only in doc0. Gold r2 (doc1): r1 is gold there too.
    assert matrix.tolist() == [[1.0, 0.5], [0.0, 1.0]]


def test_depth_f1_scores_each_level_of_the_closed_space() -> None:
    truth = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
    predicted = np.array([[1, 0, 1, 0], [0, 1, 0, 0]])
    scores = depth_f1(truth, predicted, NODES, PARENTS)
    assert scores[1] == pytest.approx(1.0)
    # Depth 2: one hit (a), one miss (b) -> P=1, R=0.5.
    assert scores[2] == pytest.approx(2 / 3)


def test_reliability_bins_compare_top1_confidence_with_accuracy() -> None:
    truth = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    scores = np.array([[0.95, 0.1], [0.9, 0.2], [0.15, 0.05], [0.3, 0.12]])
    bins = reliability_bins(truth, scores, bins=2)
    # Top-1: doc0 a (0.95, right), doc1 a (0.9, wrong), doc2 a (0.15, right), doc3 a (0.3, wrong)
    assert bins["count"].tolist() == [2, 2]
    assert bins["confidence"].tolist() == pytest.approx([0.225, 0.925])
    assert bins["accuracy"].tolist() == pytest.approx([0.5, 0.5])
