"""Validation utilities for the repository's canonical taxonomy JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_taxonomy(path: str | Path) -> dict[str, Any]:
    """Load a taxonomy document from JSON."""
    taxonomy_path = Path(path)
    with taxonomy_path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError("Taxonomy root must be a JSON object.")
    return document


def validate_taxonomy(document: dict[str, Any]) -> list[str]:
    """Return human-readable validation errors; an empty list means valid."""
    errors: list[str] = []
    labels = document.get("labels")
    if not isinstance(labels, list) or not labels:
        return ["Taxonomy must contain a non-empty 'labels' list."]

    identifiers: list[str] = []
    for position, label in enumerate(labels):
        if not isinstance(label, dict):
            errors.append(f"Label at index {position} must be an object.")
            continue
        identifier = label.get("id")
        name = label.get("name")
        if not isinstance(identifier, str) or not identifier.strip():
            errors.append(f"Label at index {position} has no valid 'id'.")
        else:
            identifiers.append(identifier)
        if not isinstance(name, str) or not name.strip():
            errors.append(f"Label at index {position} has no valid 'name'.")

    duplicates = {item for item in identifiers if identifiers.count(item) > 1}
    errors.extend(f"Duplicate label id: {item}" for item in sorted(duplicates))

    known = set(identifiers)
    parents = {
        label.get("id"): label.get("parent")
        for label in labels
        if isinstance(label, dict) and isinstance(label.get("id"), str)
    }
    for identifier, parent in parents.items():
        if parent is not None and parent not in known:
            errors.append(f"Label '{identifier}' references missing parent '{parent}'.")

    for start in known:
        seen: set[str] = set()
        current: str | None = start
        while current is not None:
            if current in seen:
                errors.append(f"Cycle detected from label '{start}'.")
                break
            seen.add(current)
            current = parents.get(current)

    return sorted(set(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a NALTRA taxonomy JSON file.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    errors = validate_taxonomy(load_taxonomy(args.path))
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"Taxonomy is valid: {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
