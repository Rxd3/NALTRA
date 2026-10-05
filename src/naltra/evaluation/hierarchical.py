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

import json
from pathlib import Path

def load_parent_child_pairs(taxonomy_path: str = "taxonomy/taxonomy.json") -> list[tuple[str, str]]:
    """Extracts (parent_id, child_id) pairs from taxonomy.json if present."""
    path = Path(taxonomy_path)
    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        pairs = []
        for label in data.get("labels", []):
            parent_id = label.get("parent")
            child_id = label.get("id")
            if parent_id is not None and child_id is not None:
                pairs.append((str(parent_id), str(child_id)))
        return pairs
    except Exception:
        return []
        