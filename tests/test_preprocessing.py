from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Add repo root and src/ to sys.path so tests can be run directly with python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from naltra.data.loader import load_jsonl, save_jsonl  # noqa: E402
from naltra.data.preprocessing import (  # noqa: E402
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
    assert len(canonical_ids) == 7
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


def test_validate_record_multilabel() -> None:
    record = {
        "id": "synthetic:1",
        "text": "Synthetic multi-label example",
        "labels": ["topic_a", "topic_b"],
        "language": "en",
        "source": "synthetic",
        "license": "TEST",
        "split": "train",
    }
    validate_record(record, allowed_labels={"topic_a", "topic_b"})


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


def test_sib200_label_mapping() -> None:
    label_map = load_label_map(PROJECT_ROOT / "taxonomy" / "label_map.json")
    canonical_ids = load_canonical_label_ids(PROJECT_ROOT / "taxonomy" / "taxonomy.json")

    raw_labels = [
        "science/technology",
        "travel",
        "politics",
        "sports",
        "health",
        "entertainment",
        "geography",
    ]

    mapped = map_labels(raw_labels, label_map)

    assert mapped == [
        "science_technology",
        "travel",
        "politics",
        "sport",
        "health",
        "arts_culture_entertainment_media",
        "geography",
    ]

    for label in mapped:
        assert label in canonical_ids


def test_multilabel_stratified_split() -> None:
    from naltra.data.splits import multilabel_stratified_split

    # Construct synthetic records with single and multi-labels
    sample_records = (
        [{"id": f"rec_{i}", "labels": ["cat_a"]} for i in range(40)]
        + [{"id": f"rec_{i+40}", "labels": ["cat_b"]} for i in range(40)]
        + [{"id": f"rec_{i+80}", "labels": ["cat_a", "cat_b"]} for i in range(20)]
    )

    train, val, test = multilabel_stratified_split(
        sample_records,
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    assert len(train) + len(val) + len(test) == 100
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15

    train_ids = {r["id"] for r in train}
    val_ids = {r["id"] for r in val}
    test_ids = {r["id"] for r in test}

    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)

    # All categories present in all splits
    for s in (train, val, test):
        labels_in_split = {lbl for r in s for lbl in r["labels"]}
        assert "cat_a" in labels_in_split
        assert "cat_b" in labels_in_split


def test_splits_same_seed_reproducibility() -> None:
    from naltra.data.splits import multilabel_stratified_split

    records = [
        {"id": f"rec_{i}", "labels": ["cat_a" if i % 2 == 0 else "cat_b"]} for i in range(30)
    ]
    train1, val1, test1 = multilabel_stratified_split(records, seed=42)
    train2, val2, test2 = multilabel_stratified_split(records, seed=42)

    assert [r["id"] for r in train1] == [r["id"] for r in train2]
    assert [r["id"] for r in val1] == [r["id"] for r in val2]
    assert [r["id"] for r in test1] == [r["id"] for r in test2]


def test_splits_different_seed_variation() -> None:
    from naltra.data.splits import multilabel_stratified_split

    # Multiple records tying on label length and frequency
    records = [{"id": f"rec_{i}", "labels": ["cat_x"]} for i in range(30)]
    train1, val1, test1 = multilabel_stratified_split(records, seed=42)
    train2, val2, test2 = multilabel_stratified_split(records, seed=999)

    # Capacities must be identical
    assert len(train1) == len(train2) == 21
    assert len(val1) == len(val2) == 5
    assert len(test1) == len(test2) == 4

    # Seed variation breaks ties differently
    assert [r["id"] for r in train1] != [r["id"] for r in train2]


def test_splits_tiny_datasets() -> None:
    from naltra.data.splits import _compute_split_capacities, multilabel_stratified_split

    # 0 records
    assert multilabel_stratified_split([]) == ([], [], [])
    assert _compute_split_capacities(0, [0.7, 0.15, 0.15]) == [0, 0, 0]

    # 1 record
    rec1 = [{"id": "r1", "labels": ["cat_a"]}]
    t1, v1, te1 = multilabel_stratified_split(rec1)
    assert len(t1) + len(v1) + len(te1) == 1
    assert _compute_split_capacities(1, [0.7, 0.15, 0.15]) == [1, 0, 0]

    # 2 records
    rec2 = [{"id": "r1", "labels": ["cat_a"]}, {"id": "r2", "labels": ["cat_b"]}]
    t2, v2, te2 = multilabel_stratified_split(rec2)
    assert len(t2) + len(v2) + len(te2) == 2
    assert _compute_split_capacities(2, [0.7, 0.15, 0.15]) == [2, 0, 0]

    generic_caps = _compute_split_capacities(101, [0.70, 0.15, 0.15])
    assert generic_caps == [71, 15, 15]


def test_splits_invalid_and_negative_ratios() -> None:
    import pytest

    from naltra.data.splits import _validate_split_ratios

    # Negative ratio
    with pytest.raises(ValueError, match="non-negative"):
        _validate_split_ratios(-0.1, 0.6, 0.5)

    # Ratios not summing to 1.0
    with pytest.raises(ValueError, match="sum to 1.0"):
        _validate_split_ratios(0.5, 0.5, 0.5)

    # Non-numeric type
    with pytest.raises(TypeError, match="numeric"):
        _validate_split_ratios("0.7", 0.15, 0.15)  # type: ignore

    # Boolean passed as ratio
    with pytest.raises(TypeError, match="numeric"):
        _validate_split_ratios(True, 0.0, 0.0)  # type: ignore

    # Non-finite ratio
    with pytest.raises(ValueError, match="finite"):
        _validate_split_ratios(float("inf"), 0.0, 0.0)

    # All-zero ratios
    with pytest.raises(ValueError, match="At least one split ratio must be positive"):
        _validate_split_ratios(0.0, 0.0, 0.0)


def test_splits_zero_test_ratio() -> None:
    from naltra.data.splits import multilabel_stratified_split, split_records

    records = [
        {"id": f"rec_{i}", "labels": ["cat_a" if i % 2 == 0 else "cat_b"]} for i in range(20)
    ]
    train, val, test = multilabel_stratified_split(
        records,
        train_ratio=0.8,
        validation_ratio=0.2,
        test_ratio=0.0,
    )
    assert len(train) == 16
    assert len(val) == 4
    assert len(test) == 0
    assert test == []

    # split_records supports test_ratio=0.0
    t_seq, v_seq, te_seq = split_records(records, train_ratio=0.8, validation_ratio=0.2)
    assert len(t_seq) == 16
    assert len(v_seq) == 4
    assert len(te_seq) == 0


def test_splits_rejects_duplicate_ids() -> None:
    import pytest

    from naltra.data.splits import multilabel_stratified_split

    records = [
        {"id": "duplicate_id", "labels": ["cat_a"]},
        {"id": "duplicate_id", "labels": ["cat_b"]},
        {"id": "unique_id", "labels": ["cat_a"]},
    ]

    with pytest.raises(
        ValueError, match="Duplicate record ID found before splitting: 'duplicate_id'"
    ):
        multilabel_stratified_split(records)


def test_compute_content_fingerprint() -> None:
    from naltra.data.preprocessing import compute_content_fingerprint

    # Normalized identical text must produce identical fingerprint
    fp1 = compute_content_fingerprint("  Hello   World!  \n")
    fp2 = compute_content_fingerprint("Hello World!")
    assert fp1 == fp2
    assert len(fp1) == 64

    # Different text must produce different fingerprint
    fp3 = compute_content_fingerprint("Hello NALTRA!")
    assert fp1 != fp3


def test_grouped_multilabel_stratified_split_prevents_leakage() -> None:
    from naltra.data.preprocessing import compute_content_fingerprint
    from naltra.data.splits import grouped_multilabel_stratified_split

    # Create 60 records where several records share identical texts under different IDs
    records = []
    for i in range(20):
        # Unique texts
        records.append(
            {
                "id": f"rec_{i}",
                "text": f"This is unique article number {i}.",
                "labels": ["cat_a" if i % 2 == 0 else "cat_b"],
            }
        )
    # Add duplicates of the first 5 articles
    for i in range(5):
        records.append(
            {
                "id": f"dup_{i}_copy1",
                "text": f"This is unique article number {i}.",
                "labels": ["cat_a" if i % 2 == 0 else "cat_b"],
            }
        )
        records.append(
            {
                "id": f"dup_{i}_copy2",
                "text": f"This is unique article number {i}.",
                "labels": ["cat_a" if i % 2 == 0 else "cat_b"],
            }
        )

    train, val, test = grouped_multilabel_stratified_split(
        records,
        group_key=lambda r: compute_content_fingerprint(r["text"]),
        train_ratio=0.7,
        validation_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    # 1. Total records preserved
    assert len(train) + len(val) + len(test) == len(records)

    # 2. Zero ID leakage
    t_ids = {r["id"] for r in train}
    v_ids = {r["id"] for r in val}
    te_ids = {r["id"] for r in test}
    assert not (t_ids & v_ids)
    assert not (t_ids & te_ids)
    assert not (v_ids & te_ids)

    # 3. Strictly zero content fingerprint leakage
    t_fps = {compute_content_fingerprint(r["text"]) for r in train}
    v_fps = {compute_content_fingerprint(r["text"]) for r in val}
    te_fps = {compute_content_fingerprint(r["text"]) for r in test}
    assert not (t_fps & v_fps)
    assert not (t_fps & te_fps)
    assert not (v_fps & te_fps)


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
    test_sib200_label_mapping()
    test_multilabel_stratified_split()
    test_splits_same_seed_reproducibility()
    test_splits_different_seed_variation()
    test_splits_tiny_datasets()
    test_splits_invalid_and_negative_ratios()
    test_splits_zero_test_ratio()
    test_splits_rejects_duplicate_ids()
    test_compute_content_fingerprint()
    test_grouped_multilabel_stratified_split_prevents_leakage()
    print("All preprocessing tests passed successfully!")
