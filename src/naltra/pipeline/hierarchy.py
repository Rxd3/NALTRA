"""Hierarchy validation and path resolution helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def validate_hierarchy(parent_by_label: Mapping[str, str | None]) -> list[str]:
    """Return errors for missing parents and cycles in a parent lookup."""
    errors: list[str] = []
    labels = set(parent_by_label)
    for label, parent in parent_by_label.items():
        if parent is not None and parent not in labels:
            errors.append(f"Label '{label}' references missing parent '{parent}'.")

    for start in labels:
        current: str | None = start
        seen: set[str] = set()
        while current is not None:
            if current in seen:
                errors.append(f"Cycle detected from label '{start}'.")
                break
            seen.add(current)
            current = parent_by_label.get(current)
    return sorted(set(errors))


def hierarchy_path(label: str, parent_by_label: Mapping[str, str | None]) -> list[str]:
    """Resolve one leaf-to-root relationship as a root-to-leaf path."""
    if label not in parent_by_label:
        raise KeyError(f"Unknown label: {label}")
    errors = validate_hierarchy(parent_by_label)
    if errors:
        raise ValueError("Invalid hierarchy: " + "; ".join(errors))

    path: list[str] = []
    current: str | None = label
    while current is not None:
        path.append(current)
        current = parent_by_label[current]
    return list(reversed(path))


def resolve_hierarchy_paths(
    labels: Iterable[str], parent_by_label: Mapping[str, str | None]
) -> list[list[str]]:
    """Resolve unique root-to-label paths for predicted labels."""
    paths = {tuple(hierarchy_path(label, parent_by_label)) for label in labels}
    return [list(path) for path in sorted(paths)]
