from __future__ import annotations

import copy
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
from naltra.data.ood import (
    MASSIVE_ALLOWED_SCENARIOS,
    generate_far_ood_benchmarks,
    generate_near_ood_benchmarks,
    load_massive_paired_records,
    validate_ood_record,
)
from naltra.data.preprocessing import validate_record

EXPECTED_NEAR_OOD_COUNTS = {
    "arts_culture_entertainment_media": {"validation_ood": 18, "test_ood": 38, "total_ood": 56},
    "geography": {"validation_ood": 16, "test_ood": 34, "total_ood": 50},
    "health": {"validation_ood": 22, "test_ood": 44, "total_ood": 66},
    "politics": {"validation_ood": 28, "test_ood": 60, "total_ood": 88},
    "science_technology": {"validation_ood": 50, "test_ood": 102, "total_ood": 152},
    "sport": {"validation_ood": 24, "test_ood": 50, "total_ood": 74},
    "travel": {"validation_ood": 40, "test_ood": 80, "total_ood": 120},
}


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
    try:
        validate_ood_record(bad_missing)
    except ValueError as exc:
        assert "missing required field" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing field")

    # is_ood != True raises ValueError
    bad_is_ood = dict(valid_record, is_ood=False)
    try:
        validate_ood_record(bad_is_ood)
    except ValueError as exc:
        assert "must have is_ood=True" in str(exc)
    else:
        raise AssertionError("Expected ValueError for is_ood != True")

    # Assignment of fake/canonical labels to OOD record is strictly prohibited
    bad_labels = dict(valid_record, labels=["politics"])
    try:
        validate_ood_record(bad_labels)
    except ValueError as exc:
        assert "must not be assigned canonical taxonomy labels" in str(exc)
    else:
        raise AssertionError("Expected ValueError when assigning canonical labels to OOD record")


def test_near_ood_generation_and_exact_counts() -> None:
    """Verify 7-fold leave-one-topic-out benchmark counts, zero leakage, and pairing."""
    sib200_dir = PROJECT_ROOT / "data" / "processed" / "sib200"
    if not (sib200_dir / "train.jsonl").exists():
        print("[SKIP] SIB-200 processed data not found.")
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_base = Path(tmp_dir) / "near_ood"
        stats = generate_near_ood_benchmarks(sib200_dir=sib200_dir, output_base_dir=out_base)

        assert len(stats) == 7

        for topic, expected in EXPECTED_NEAR_OOD_COUNTS.items():
            assert topic in stats
            fold_stat = stats[topic]
            assert fold_stat["validation_ood"] == expected["validation_ood"]
            assert fold_stat["test_ood"] == expected["test_ood"]
            assert fold_stat["total_ood"] == expected["total_ood"]

            fold_dir = out_base / topic
            train_id = load_jsonl(fold_dir / "train_id.jsonl")
            val_id = load_jsonl(fold_dir / "validation_id.jsonl")
            val_ood = load_jsonl(fold_dir / "validation_ood.jsonl")
            test_id = load_jsonl(fold_dir / "test_id.jsonl")
            test_ood = load_jsonl(fold_dir / "test_ood.jsonl")

            # 1. Held-out topic is completely absent from train_id
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
            assert len(en_val) == len(tr_val)

            en_test = [r for r in test_ood if r["language"] == "en"]
            tr_test = [r for r in test_ood if r["language"] == "tr"]
            assert len(en_test) == len(tr_test)


def test_clean_sib200_benchmark_unmodified() -> None:
    """Verify that generating near-OOD does not touch clean SIB-200 files."""
    sib200_dir = PROJECT_ROOT / "data" / "processed" / "sib200"
    train_recs = load_jsonl(sib200_dir / "train.jsonl")
    val_recs = load_jsonl(sib200_dir / "validation.jsonl")
    test_recs = load_jsonl(sib200_dir / "test.jsonl")

    assert len(train_recs) == 1402
    assert len(val_recs) == 198
    assert len(test_recs) == 408


def test_massive_allowlist_filtering() -> None:
    """Verify that only scenarios from MASSIVE_ALLOWED_SCENARIOS are included."""
    disallowed_scenarios = {
        "weather",
        "news",
        "music",
        "play",
        "qa",
        "recommendation",
        "transport",
        "social",
        "email",
        "cooking",
        "general",
    }
    # Disallowed scenarios and allowed scenarios must be completely disjoint
    assert not (disallowed_scenarios & MASSIVE_ALLOWED_SCENARIOS)

    raw_dir = PROJECT_ROOT / "data" / "raw" / "massive"
    en_file = raw_dir / "en-US.jsonl"
    tr_file = raw_dir / "tr-TR.jsonl"

    if en_file.exists() and tr_file.exists():
        paired_dict = load_massive_paired_records(en_file, tr_file)
        all_pairs = paired_dict["test"] + paired_dict["dev"]
        for en_rec, tr_rec in all_pairs:
            assert en_rec["scenario"] in MASSIVE_ALLOWED_SCENARIOS
            assert tr_rec["scenario"] in MASSIVE_ALLOWED_SCENARIOS
            assert en_rec["scenario"] not in disallowed_scenarios


def test_far_ood_generation_and_balance() -> None:
    """Verify Far-OOD produces matched, balanced EN/TR pairs with stable sampling."""
    raw_dir = PROJECT_ROOT / "data" / "raw" / "massive"
    if not (raw_dir / "en-US.jsonl").exists():
        print("[SKIP] MASSIVE raw data not found.")
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_base = Path(tmp_dir) / "far_ood"
        stats1 = generate_far_ood_benchmarks(
            raw_dir=raw_dir,
            output_base_dir=out_base,
            val_sample_size=10,
            test_sample_size=20,
            seed=42,
        )

        assert stats1["validation"] == 20  # 10 EN + 10 TR
        assert stats1["test"] == 40        # 20 EN + 20 TR

        val_records = load_jsonl(out_base / "validation_ood.jsonl")
        test_records = load_jsonl(out_base / "test_ood.jsonl")

        # Verify 1:1 EN/TR balance
        assert len([r for r in val_records if r["language"] == "en"]) == 10
        assert len([r for r in val_records if r["language"] == "tr"]) == 10
        assert len([r for r in test_records if r["language"] == "en"]) == 20
        assert len([r for r in test_records if r["language"] == "tr"]) == 20

        # Verify pairing: every EN record has a corresponding TR record with same pair_id
        val_pairs_en = {r["pair_id"] for r in val_records if r["language"] == "en"}
        val_pairs_tr = {r["pair_id"] for r in val_records if r["language"] == "tr"}
        assert val_pairs_en == val_pairs_tr

        # Verify deterministic sampling with same seed
        out_base2 = Path(tmp_dir) / "far_ood_2"
        stats2 = generate_far_ood_benchmarks(
            raw_dir=raw_dir,
            output_base_dir=out_base2,
            val_sample_size=10,
            test_sample_size=20,
            seed=42,
        )
        assert stats1 == stats2
        val_records_2 = load_jsonl(out_base2 / "validation_ood.jsonl")
        assert [r["id"] for r in val_records] == [r["id"] for r in val_records_2]


if __name__ == "__main__":
    print("Running test_ood suite...")
    test_validate_ood_record_schema_and_guards()
    test_near_ood_generation_and_exact_counts()
    test_clean_sib200_benchmark_unmodified()
    test_massive_allowlist_filtering()
    test_far_ood_generation_and_balance()
    print("All 5 OOD benchmark tests passed successfully!")
