import pytest

from naltra.pipeline.hierarchy import (
    hierarchy_path,
    resolve_hierarchy_paths,
    validate_hierarchy,
)

PARENTS = {
    "Technology": None,
    "Hardware": "Technology",
    "Semiconductors": "Hardware",
    "Software": "Technology",
}


def test_hierarchy_path_is_root_to_leaf() -> None:
    assert hierarchy_path("Semiconductors", PARENTS) == [
        "Technology",
        "Hardware",
        "Semiconductors",
    ]


def test_resolve_hierarchy_paths_returns_unique_paths() -> None:
    assert resolve_hierarchy_paths(["Hardware", "Hardware"], PARENTS) == [
        ["Technology", "Hardware"]
    ]


def test_hierarchy_validation_reports_missing_parent() -> None:
    assert validate_hierarchy({"child": "missing"}) == [
        "Label 'child' references missing parent 'missing'."
    ]


def test_unknown_label_is_rejected() -> None:
    with pytest.raises(KeyError, match="Unknown label"):
        hierarchy_path("Unknown", PARENTS)
