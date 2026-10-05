from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# Add repo root and src/ to sys.path so tests can be run directly with python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from naltra.data.loader import load_jsonl, save_jsonl  # noqa: E402
from naltra.data.ood import (  # noqa: E402
    generate_near_ood_benchmarks,
    validate_ood_record,
    validate_sib200_source_data,
)
from naltra.data.preprocessing import validate_record  # noqa: E402

SIB200_CANONICAL_TOPICS = [
    "arts_culture_entertainment_media",
    "geography",
    "health",
    "politics",
    "science_technology",
    "sport",
    "travel",
]

EXPECTED_NEAR_OOD_COUNTS = {
    "arts_culture_entertainment_media": {"validation_ood": 18, "test_ood": 38, "total_ood": 56},
    "geography": {"validation_ood": 16, "test_ood": 34, "total_ood": 50},
    "health": {"validation_ood": 22, "test_ood": 44, "total_ood": 66},
    "politics": {"validation_ood": 28, "test_ood": 60, "total_ood": 88},
    "science_technology": {"validation_ood": 50, "test_ood": 102, "total_ood": 152},
    "sport": {"validation_ood": 24, "test_ood": 50, "total_ood": 74},
    "travel": {"validation_ood": 40, "test_ood": 80, "total_ood": 120},
}


def create_synthetic_sib200_dataset(base_dir: Path) -> Path:
    """Create compliant synthetic SIB-200 dataset with 7 topics and paired EN/TR records."""
    base_dir.mkdir(parents=True, exist_ok=True)
    split_configs = {
        "train": 2,  # 2 pairs * 7 topics * 2 langs = 28 records
        "validation": 1,  # 1 pair * 7 topics * 2 langs = 14 records
        "test": 2,  # 2 pairs * 7 topics * 2 langs = 28 records
    }

    for split_name, num_pairs in split_configs.items():
        records: list[dict[str, Any]] = []
        for topic in SIB200_CANONICAL_TOPICS:
            for idx in range(num_pairs):
                pair_id = f"sib200:{topic}_{split_name}_{idx}"
                for lang in ("en", "tr"):
                    rec = {
                        "id": f"sib200:{lang}:{topic}_{split_name}_{idx}",
                        "text": f"Synthetic {lang} text for topic {topic} index {idx}.",
                        "labels": [topic],
                        "language": lang,
                        "source": "sib200",
                        "license": "CC BY-SA 4.0",
                        "split": split_name,
                        "source_labels": [topic],
                        "source_id": f"{topic}_{split_name}_{idx}",
                        "pair_id": pair_id,
                    }
                    validate_record(rec)
                    records.append(rec)
        save_jsonl(records, base_dir / f"{split_name}.jsonl")

    return base_dir


def test_validate_ood_record_schema_and_guards() -> None:
    """Verify OOD schema validator enforces required fields, flags, and rejects canonical labels."""
    valid_record = {
        "id": "sib200:ood:politics:validation:sib200:en:1322",
        "text": "Sample text for OOD testing.",
        "language": "en",
        "source": "sib200",
        "license": "CC BY-SA 4.0",
        "split": "validation",
        "is_ood": True,
        "ood_type": "near_ood",
        "ood_source": "sib200_heldout",
        "ood_reason": "heldout_topic:politics",
        "source_id": "1322",
        "source_label": "politics",
        "pair_id": "sib200:1322",
    }
    # Valid record passes
    validate_ood_record(valid_record)

    # Empty list for labels is allowed
    record_empty_labels = dict(valid_record, labels=[])
    validate_ood_record(record_empty_labels)

    # Missing required field raises ValueError
    bad_missing = dict(valid_record)
    del bad_missing["ood_source"]
    with pytest.raises(ValueError, match="missing required field"):
        validate_ood_record(bad_missing)

    # is_ood != True raises ValueError
    bad_is_ood = dict(valid_record, is_ood=False)
    with pytest.raises(ValueError, match="must have is_ood=True"):
        validate_ood_record(bad_is_ood)

    # Assignment of fake/canonical labels to OOD record is strictly prohibited
    bad_labels = dict(valid_record, labels=["politics"])
    with pytest.raises(ValueError, match="must not be assigned canonical taxonomy labels"):
        validate_ood_record(bad_labels)


def test_validate_ood_record_strict_types() -> None:
    valid_record = {
        "id": "sib200:ood:politics:validation:sib200:en:1322",
        "text": "Sample text for OOD testing.",
        "language": "en",
        "source": "sib200",
        "license": "CC BY-SA 4.0",
        "split": "validation",
        "is_ood": True,
        "ood_type": "near_ood",
        "ood_source": "sib200_heldout",
        "ood_reason": "heldout_topic:politics",
        "source_id": "1322",
        "source_label": "politics",
        "pair_id": "sib200:1322",
    }

    # numeric license
    with pytest.raises(ValueError, match="field 'license' must be a non-empty string"):
        validate_ood_record(dict(valid_record, license=123))

    # empty license
    with pytest.raises(ValueError, match="field 'license' must be a non-empty string"):
        validate_ood_record(dict(valid_record, license="   "))

    # empty pair_id
    with pytest.raises(ValueError, match="field 'pair_id' must be a non-empty string"):
        validate_ood_record(dict(valid_record, pair_id=""))

    # numeric pair_id
    with pytest.raises(ValueError, match="field 'pair_id' must be a non-empty string"):
        validate_ood_record(dict(valid_record, pair_id=42))

    # empty source
    with pytest.raises(ValueError, match="field 'source' must be a non-empty string"):
        validate_ood_record(dict(valid_record, source="  "))

    # invalid source_id (bool)
    with pytest.raises(ValueError, match="must be a valid non-empty scalar identifier"):
        validate_ood_record(dict(valid_record, source_id=True))

    # invalid source_id (empty string)
    with pytest.raises(ValueError, match="must be a valid non-empty scalar identifier"):
        validate_ood_record(dict(valid_record, source_id="   "))

    # invalid split
    with pytest.raises(ValueError, match="has invalid split"):
        validate_ood_record(dict(valid_record, split="invalid_split"))

    # invalid language
    with pytest.raises(ValueError, match="has invalid language"):
        validate_ood_record(dict(valid_record, language="fr"))

    # invalid ood_type
    with pytest.raises(ValueError, match="has invalid ood_type"):
        validate_ood_record(dict(valid_record, ood_type="unknown_ood"))


def test_validate_sib200_source_data_regressions() -> None:
    valid_train = [
        {
            "id": "sib:en:1",
            "text": "English text 1",
            "labels": ["politics"],
            "language": "en",
            "source": "sib200",
            "license": "CC BY-SA 4.0",
            "split": "train",
            "pair_id": "pair:1",
        },
        {
            "id": "sib:tr:1",
            "text": "Turkish text 1",
            "labels": ["politics"],
            "language": "tr",
            "source": "sib200",
            "license": "CC BY-SA 4.0",
            "split": "train",
            "pair_id": "pair:1",
        },
    ]
    valid_val = [
        {
            "id": "sib:en:2",
            "text": "English text 2",
            "labels": ["sport"],
            "language": "en",
            "source": "sib200",
            "license": "CC BY-SA 4.0",
            "split": "validation",
            "pair_id": "pair:2",
        },
        {
            "id": "sib:tr:2",
            "text": "Turkish text 2",
            "labels": ["sport"],
            "language": "tr",
            "source": "sib200",
            "license": "CC BY-SA 4.0",
            "split": "validation",
            "pair_id": "pair:2",
        },
    ]
    valid_test = [
        {
            "id": "sib:en:3",
            "text": "English text 3",
            "labels": ["health"],
            "language": "en",
            "source": "sib200",
            "license": "CC BY-SA 4.0",
            "split": "test",
            "pair_id": "pair:3",
        },
        {
            "id": "sib:tr:3",
            "text": "Turkish text 3",
            "labels": ["health"],
            "language": "tr",
            "source": "sib200",
            "license": "CC BY-SA 4.0",
            "split": "test",
            "pair_id": "pair:3",
        },
    ]

    # Valid baseline passes
    validate_sib200_source_data(valid_train, valid_val, valid_test)

    # 1. Duplicate EN in one pair
    bad_pair_dup_en = [
        dict(valid_train[0]),
        dict(valid_train[0], id="sib:en:1b"),
    ]
    with pytest.raises(ValueError, match="must have exactly one 'en' and one 'tr' record"):
        validate_sib200_source_data(bad_pair_dup_en, valid_val, valid_test)

    # 2. Missing TR record (single record in pair)
    bad_missing_tr = [dict(valid_train[0])]
    with pytest.raises(ValueError, match="expected exactly 2"):
        validate_sib200_source_data(bad_missing_tr, valid_val, valid_test)

    # 3. Pair split mismatch
    bad_split_mismatch = [
        dict(valid_train[0]),
        dict(valid_train[1], split="validation"),
    ]
    with pytest.raises(ValueError, match="mismatched splits"):
        validate_sib200_source_data(bad_split_mismatch, valid_val, valid_test)

    # 4. Pair label mismatch
    bad_label_mismatch = [
        dict(valid_train[0]),
        dict(valid_train[1], labels=["travel"]),
    ]
    with pytest.raises(ValueError, match="mismatched labels"):
        validate_sib200_source_data(bad_label_mismatch, valid_val, valid_test)

    # 5. pair_id reused across train and validation
    bad_val_reused_pair = [
        dict(valid_val[0], pair_id="pair:1"),
        dict(valid_val[1], pair_id="pair:1"),
    ]
    with pytest.raises(ValueError, match="SIB-200 pair IDs overlap between train and validation"):
        validate_sib200_source_data(valid_train, bad_val_reused_pair, valid_test)

    # 6. Duplicate record IDs
    bad_dup_id = [
        dict(valid_train[0]),
        dict(valid_train[1], id=valid_train[0]["id"]),
    ]
    with pytest.raises(ValueError, match="Duplicate SIB-200 record ID detected"):
        validate_sib200_source_data(bad_dup_id, valid_val, valid_test)


def test_missing_input_preflight_checks(tmp_path: Path) -> None:
    from naltra.data.code_switching import generate_code_switch_benchmarks
    from naltra.data.noise import generate_noisy_benchmarks
    from naltra.data.ood import generate_near_ood_benchmarks

    empty_dir = tmp_path / "non_existent"

    # Noise generation preflight
    with pytest.raises(FileNotFoundError, match="Cannot generate noisy benchmarks"):
        generate_noisy_benchmarks(
            processed_base_dir=empty_dir,
            output_base_dir=tmp_path / "noisy_out",
        )

    # Code-switch generation preflight
    with pytest.raises(FileNotFoundError, match="Cannot generate code-switch benchmarks"):
        generate_code_switch_benchmarks(
            processed_base_dir=empty_dir,
            output_base_dir=tmp_path / "cs_out",
        )

    # Near-OOD generation preflight
    with pytest.raises(FileNotFoundError, match="Cannot generate Near-OOD benchmarks"):
        generate_near_ood_benchmarks(
            sib200_dir=empty_dir,
            output_base_dir=tmp_path / "ood_out",
        )


def test_near_ood_generation_synthetic(tmp_path: Path) -> None:
    """Verify Near-OOD generation on synthetic fixture: 7 folds, zero leakage, 1:1 balance."""
    sib_dir = tmp_path / "clean_sib200"
    create_synthetic_sib200_dataset(sib_dir)

    out_base = tmp_path / "near_ood"
    stats = generate_near_ood_benchmarks(sib200_dir=sib_dir, output_base_dir=out_base)

    assert len(stats) == 7

    for topic in SIB200_CANONICAL_TOPICS:
        assert topic in stats
        fold_stat = stats[topic]
        # In our fixture: 1 pair in val (2 records), 2 pairs in test (4 records)
        assert fold_stat["validation_ood"] == 2
        assert fold_stat["test_ood"] == 4
        assert fold_stat["total_ood"] == 6

        fold_dir = out_base / topic
        train_id = load_jsonl(fold_dir / "train_id.jsonl")
        val_id = load_jsonl(fold_dir / "validation_id.jsonl")
        val_ood = load_jsonl(fold_dir / "validation_ood.jsonl")
        test_id = load_jsonl(fold_dir / "test_id.jsonl")
        test_ood = load_jsonl(fold_dir / "test_ood.jsonl")

        assert len(val_id) > 0
        assert len(test_id) > 0

        # 1. Held-out topic is absent from train_id
        for r in train_id:
            assert r["labels"][0] != topic

        # 2. Train contains exactly 6 unique topics
        train_topics = {r["labels"][0] for r in train_id}
        assert len(train_topics) == 6
        assert topic not in train_topics

        # 3. Near-OOD evaluation contains only the held-out topic
        for r in val_ood + test_ood:
            assert r["source_label"] == topic
            assert r["is_ood"] is True
            assert r["ood_type"] == "near_ood"

        # 4. Zero pair leakage between ID train and OOD
        train_pairs = {r["pair_id"] for r in train_id}
        ood_pairs = {r["pair_id"] for r in val_ood + test_ood}
        assert not (train_pairs & ood_pairs)

        # 5. EN/TR pairing preserved in OOD
        en_val = [r for r in val_ood if r["language"] == "en"]
        tr_val = [r for r in val_ood if r["language"] == "tr"]
        assert len(en_val) == len(tr_val) == 1

        en_test = [r for r in test_ood if r["language"] == "en"]
        tr_test = [r for r in test_ood if r["language"] == "tr"]
        assert len(en_test) == len(tr_test) == 2


def test_clean_sib200_benchmark_unmodified(tmp_path: Path) -> None:
    """Verify that generating near-OOD benchmarks leaves the clean SIB-200 files exactly unchanged.

    Follows the 5 required review steps:
    1. creates a synthetic SIB-200 fixture,
    2. saves the original clean files,
    3. runs generate_near_ood_benchmarks(),
    4. reloads the clean files,
    5. verifies that their contents are exactly unchanged.
    """
    # 1. Create a synthetic SIB-200 fixture
    sib_dir = tmp_path / "clean_sib200"
    create_synthetic_sib200_dataset(sib_dir)

    # 2. Save the original clean files
    original_contents: dict[str, bytes] = {}
    original_records: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "validation", "test"):
        file_path = sib_dir / f"{split}.jsonl"
        original_contents[split] = file_path.read_bytes()
        original_records[split] = load_jsonl(file_path)

    # 3. Run generate_near_ood_benchmarks()
    out_base = tmp_path / "near_ood_out"
    generate_near_ood_benchmarks(sib200_dir=sib_dir, output_base_dir=out_base)

    # 4. Reload the clean files
    # 5. Verify that their contents are exactly unchanged
    for split in ("train", "validation", "test"):
        file_path = sib_dir / f"{split}.jsonl"
        current_bytes = file_path.read_bytes()
        assert (
            current_bytes == original_contents[split]
        ), f"Clean SIB-200 file {split}.jsonl was modified!"
        current_records = load_jsonl(file_path)
        assert current_records == original_records[split]


# Optional integration tests requiring real local datasets


def test_real_sib200_near_ood_integration(tmp_path: Path) -> None:
    """Optional integration test on real processed SIB-200 benchmark files."""
    sib200_dir = PROJECT_ROOT / "data" / "processed" / "sib200"
    if not (sib200_dir / "train.jsonl").exists():
        pytest.skip("Local processed SIB-200 dataset not found.")

    out_base = tmp_path / "near_ood_real"
    stats = generate_near_ood_benchmarks(sib200_dir=sib200_dir, output_base_dir=out_base)

    assert len(stats) == 7
    for topic, expected in EXPECTED_NEAR_OOD_COUNTS.items():
        assert topic in stats
        fold_stat = stats[topic]
        assert fold_stat["validation_ood"] == expected["validation_ood"]
        assert fold_stat["test_ood"] == expected["test_ood"]
        assert fold_stat["total_ood"] == expected["total_ood"]
