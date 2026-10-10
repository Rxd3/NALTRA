"""Hierarchy-aware label expansion."""

from collections.abc import Mapping

from naltra.pipeline.hierarchy import hierarchy_path


def expand_with_ancestors(labels: set[str], parent_by_label: Mapping[str, str | None]) -> set[str]:
    """Add all ancestors for each label."""
    expanded: set[str] = set()
    for label in labels:
        expanded.update(hierarchy_path(label, parent_by_label))
    return expanded
