"""Hierarchy-aware label expansion and metrics."""

from collections.abc import Mapping

from naltra.evaluation.metrics import precision_recall_f1
from naltra.pipeline.hierarchy import hierarchy_path


def expand_with_ancestors(labels: set[str], parent_by_label: Mapping[str, str | None]) -> set[str]:
    """Add all ancestors for each label."""
    expanded: set[str] = set()
    for label in labels:
        expanded.update(hierarchy_path(label, parent_by_label))
    return expanded


def hierarchical_precision_recall_f1(
    true_labels: set[str],
    predicted_labels: set[str],
    parent_by_label: Mapping[str, str | None],
) -> tuple[float, float, float]:
    """Score sets after expanding each label to its ancestors."""
    return precision_recall_f1(
        expand_with_ancestors(true_labels, parent_by_label),
        expand_with_ancestors(predicted_labels, parent_by_label),
    )
