from __future__ import annotations

from pathlib import Path
import sys
import tempfile

# Add repo root and src/ to sys.path so tests can be run directly with python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from naltra.data.loader import load_jsonl, save_jsonl
from naltra.data.preprocessing import (
    load_canonical_label_ids,
    load_label_map,
    map_labels,
    preprocess_record_text,
    validate_record,
)


def test_load_label_map_and_canonical_ids() -> None:
    label_map = load_label_map(PROJECT_ROOT / "taxonomy" / "label_map.json")
    canonical_ids = load_canonical_label_ids(PROJECT_ROOT / "taxonomy" / "taxonomy.json")

    assert isinstance(label_map, dict)
    assert len(label_map) > 0
    assert len(canonical_ids) == 151
    for target in label_map.values():
        assert target in canonical_ids


def test_map_labels_success_and_deduplication() -> None:
    label_map = {"sports": "sport", "politics": "politics"}
    result = map_labels(["sports", "politics", "sports"], label_map)
    assert result == ["sport", "politics"]


def test_map_labels_unknown_label_raises_key_error() -> None:
    label_map = {"sports": "sport"}
    try:
        map_labels(["unmapped_unknown_category"], label_map)
    except KeyError as exc:
        assert "Unknown source label" in str(exc)
    else:
        raise AssertionError("Expected KeyError for unknown label")


def test_normalize_record_text() -> None:
    raw = "  Bu bir   Türkçe    cümledir.\n\nİkinci satır. "
    normalized = preprocess_record_text(raw)
    assert normalized == "Bu bir Türkçe cümledir. İkinci satır."


def test_validate_record_valid() -> None:
    record = {
        "id": "sib200:en:0",
        "text": "Sample text",
        "labels": ["science_technology"],
        "language": "en",
        "source": "sib200",
        "license": "CC BY-SA 4.0",
        "split": "train",
    }
    validate_record(record, allowed_labels={"science_technology"})


def test_validate_record_missing_field() -> None:
    record = {
        "id": "sib200:en:0",
        "text": "Sample text",
        "labels": ["science_technology"],
    }
    try:
        validate_record(record)
    except ValueError as exc:
        assert "missing required field" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing field")


def test_validate_record_invalid_language() -> None:
    record = {
        "id": "sib200:en:0",
        "text": "Sample text",
        "labels": ["science_technology"],
        "language": "french",
        "source": "sib200",
        "license": "CC BY-SA 4.0",
        "split": "train",
    }
    try:
        validate_record(record)
    except ValueError as exc:
        assert "invalid language" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid language")


def test_validate_record_unknown_canonical_label() -> None:
    record = {
        "id": "sib200:en:0",
        "text": "Sample text",
        "labels": ["non_existent_label"],
        "language": "en",
        "source": "sib200",
        "license": "CC BY-SA 4.0",
        "split": "train",
    }
    try:
        validate_record(record, allowed_labels={"science_technology", "sport"})
    except ValueError as exc:
        assert "not in canonical taxonomy" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown canonical label")


def test_save_and_load_jsonl_roundtrip() -> None:
    sample_records = [
        {
            "id": "sib200:tr:1",
            "text": "Türkçe özel karakterler: ğüşöçıİ test.",
            "labels": ["science_technology"],
            "language": "tr",
            "source": "sib200",
            "license": "CC BY-SA 4.0",
            "split": "train",
            "pair_id": "sib200:1",
        }
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test_split.jsonl"
        save_jsonl(sample_records, file_path)
        loaded = load_jsonl(file_path)

        assert loaded == sample_records
        assert loaded[0]["text"] == "Türkçe özel karakterler: ğüşöçıİ test."


def test_validate_record_multilabel() -> None:
    record = {
        "id": "multifin:Israel-4145",
        "text": "Revenue Recognition and corporate tax audit",
        "labels": ["accounting_assurance", "tax"],
        "language": "en",
        "source": "multifin",
        "license": "CC BY-NC 4.0",
        "split": "train",
        "source_id": "Israel-4145",
        "source_labels": ["Accounting & Assurance", "Tax"],
    }
    validate_record(record, allowed_labels={"accounting_assurance", "tax"})


def test_multifin_label_mapping() -> None:
    label_map = load_label_map(PROJECT_ROOT / "taxonomy" / "label_map.json")
    canonical_ids = load_canonical_label_ids(PROJECT_ROOT / "taxonomy" / "taxonomy.json")

    # Real MultiFin multi-label combination
    raw_labels = ["Accounting & Assurance", "Tax", "VAT & Customs"]
    mapped = map_labels(raw_labels, label_map)

    assert mapped == ["accounting_assurance", "tax", "vat_customs"]
    for l in mapped:
        assert l in canonical_ids


if __name__ == "__main__":
    print("Running test_preprocessing suite...")
    test_load_label_map_and_canonical_ids()
    test_map_labels_success_and_deduplication()
    test_map_labels_unknown_label_raises_key_error()
    test_normalize_record_text()
    test_validate_record_valid()
    test_validate_record_multilabel()
    test_validate_record_missing_field()
    test_validate_record_invalid_language()
    test_validate_record_unknown_canonical_label()
    test_save_and_load_jsonl_roundtrip()
    test_multifin_label_mapping()
    print("All 11 preprocessing tests passed successfully!")
