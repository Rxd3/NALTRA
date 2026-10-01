import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PATH = PROJECT_ROOT / "taxonomy" / "validation.py"

spec = importlib.util.spec_from_file_location(
    "taxonomy_validation",
    VALIDATION_PATH,
)

if spec is None or spec.loader is None:
    raise ImportError(
        f"Could not load taxonomy validation from {VALIDATION_PATH}"
    )

taxonomy_validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(taxonomy_validation)

load_taxonomy = taxonomy_validation.load_taxonomy
validate_taxonomy = taxonomy_validation.validate_taxonomy


def test_repository_taxonomy_loads_and_is_valid() -> None:
    document = load_taxonomy(
        PROJECT_ROOT / "taxonomy" / "taxonomy.json"
    )

    assert document["version"] == "0.2.0"
    assert validate_taxonomy(document) == []
    assert len(document["labels"]) == 151
    assert {label["id"] for label in document["labels"]} >= {
        "science_technology",
        "politics",
        "sport",
    }


def test_taxonomy_validation_detects_a_cycle() -> None:
    document = {
        "labels": [
            {"id": "a", "name": "A", "parent": "b"},
            {"id": "b", "name": "B", "parent": "a"},
        ]
    }

    assert any(
        "Cycle" in error
        for error in validate_taxonomy(document)
    )


if __name__ == "__main__":
    print("Running test_taxonomy suite...")
    test_repository_taxonomy_loads_and_is_valid()
    test_taxonomy_validation_detects_a_cycle()
    print("All taxonomy tests passed successfully!")