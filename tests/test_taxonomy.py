from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PATH = PROJECT_ROOT / "taxonomy" / "validation.py"

spec = importlib.util.spec_from_file_location(
    "taxonomy_validation",
    VALIDATION_PATH,
)

if spec is None or spec.loader is None:
    raise ImportError(f"Could not load taxonomy validation from {VALIDATION_PATH}")

taxonomy_validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(taxonomy_validation)

load_taxonomy = taxonomy_validation.load_taxonomy
validate_taxonomy = taxonomy_validation.validate_taxonomy


def test_repository_taxonomy_loads_and_is_valid() -> None:
    document = load_taxonomy(PROJECT_ROOT / "taxonomy" / "taxonomy.json")

    assert document["version"] == "0.3.0"
    assert validate_taxonomy(document) == []
    assert len(document["labels"]) == 7
    assert {label["id"] for label in document["labels"]} == {
        "science_technology",
        "travel",
        "politics",
        "sport",
        "health",
        "arts_culture_entertainment_media",
        "geography",
    }


def test_repository_label_map_targets_are_canonical() -> None:
    document = load_taxonomy(PROJECT_ROOT / "taxonomy" / "taxonomy.json")
    canonical_ids = {label["id"] for label in document["labels"]}
    label_map_path = PROJECT_ROOT / "taxonomy" / "label_map.json"

    import json

    with open(label_map_path, encoding="utf-8") as f:
        mapping = json.load(f)

    aliases = mapping.get("aliases", {})
    assert aliases, "Expected non-empty 'aliases' in label_map.json"
    for _src_label, target_label in aliases.items():
        assert target_label in canonical_ids, f"Target '{target_label}' not in taxonomy"
