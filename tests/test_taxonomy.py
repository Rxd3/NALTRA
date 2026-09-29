from pathlib import Path

from taxonomy.validation import load_taxonomy, validate_taxonomy

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_repository_taxonomy_loads_and_is_valid() -> None:
    document = load_taxonomy(PROJECT_ROOT / "taxonomy" / "taxonomy.json")

    assert document["version"] == "0.1.0"
    assert validate_taxonomy(document) == []
    assert {label["id"] for label in document["labels"]} >= {
        "technology",
        "hardware",
        "semiconductors",
    }


def test_taxonomy_validation_detects_a_cycle() -> None:
    document = {
        "labels": [
            {"id": "a", "name": "A", "parent": "b"},
            {"id": "b", "name": "B", "parent": "a"},
        ]
    }

    assert any("Cycle" in error for error in validate_taxonomy(document))
