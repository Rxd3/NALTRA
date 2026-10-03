"""Out-of-Distribution (OOD) benchmark generation for NALTRA.

Supports two separate, complementary OOD evaluation settings:
1. Near-OOD: SIB-200 Leave-One-Topic-Out (7 deterministic topic folds).
2. Far-OOD: Amazon MASSIVE dataset (voice-assistant commands filtered to non-news domains).
"""

from __future__ import annotations

from collections import Counter
import copy
import hashlib
from pathlib import Path
import random
import tarfile
from typing import Any, Sequence
import urllib.request

from naltra.data.loader import load_jsonl, save_jsonl
from naltra.data.preprocessing import validate_record

MASSIVE_TAR_URL = (
    "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
)

# Explicit allowlist of voice-assistant command scenarios that do not overlap SIB-200 news topics
MASSIVE_ALLOWED_SCENARIOS: set[str] = {
    "alarm",      # alarm_set, alarm_query, alarm_remove
    "datetime",   # datetime_query, datetime_convert
    "calendar",   # calendar_set, calendar_query, calendar_remove
    "lists",      # lists_createoradd, lists_query, lists_remove
    "iot",        # iot_hue_lighton, iot_hue_lightoff, iot_coffee, iot_wemo_on, etc.
    "audio",      # audio_volume_up, audio_volume_down, audio_volume_mute, audio_volume_other
    "takeaway",   # takeaway_order, takeaway_query
}

# Excluded scenarios with potential semantic overlap with taxonomy or news topics:
# weather, news, music, play, qa, recommendation, transport, social, email, cooking, general

REQUIRED_OOD_FIELDS = (
    "id",
    "text",
    "language",
    "source",
    "license",
    "split",
    "is_ood",
    "ood_type",
    "ood_source",
    "ood_reason",
    "source_id",
    "source_label",
    "pair_id",
)


def validate_ood_record(record: dict[str, Any]) -> None:
    """Validate that an OOD record complies with the dedicated NALTRA OOD schema."""
    if not isinstance(record, dict):
        raise ValueError("OOD record must be a dictionary.")

    for field in REQUIRED_OOD_FIELDS:
        if field not in record:
            raise ValueError(f"OOD record '{record.get('id', 'unknown')}' is missing required field '{field}'.")

    if not isinstance(record["id"], str) or not record["id"].strip():
        raise ValueError("OOD record 'id' must be a non-empty string.")

    if not isinstance(record["text"], str) or not record["text"].strip():
        raise ValueError(f"OOD record '{record['id']}' has empty or non-string 'text'.")

    if record["language"] not in ("en", "tr"):
        raise ValueError(f"OOD record '{record['id']}' has unsupported language '{record['language']}'.")

    if record["split"] not in ("validation", "test"):
        raise ValueError(f"OOD record '{record['id']}' has invalid split '{record['split']}'.")

    if record["is_ood"] is not True:
        raise ValueError(f"OOD record '{record['id']}' must have is_ood=True.")

    if record["ood_type"] not in ("near_ood", "far_ood"):
        raise ValueError(f"OOD record '{record['id']}' has invalid ood_type '{record['ood_type']}'.")

    # Critical requirement: OOD records must NOT contain non-empty canonical labels
    if "labels" in record and record["labels"]:
        raise ValueError(
            f"OOD record '{record['id']}' must not be assigned canonical taxonomy labels. "
            f"Got: {record['labels']}"
        )


def generate_near_ood_benchmarks(
    sib200_dir: str | Path = "data/processed/sib200",
    output_base_dir: str | Path = "data/ood/near/sib200",
) -> dict[str, dict[str, int]]:
    """Generate 7-fold leave-one-topic-out Near-OOD benchmarks from SIB-200.

    For each topic, creates:
      - train_id.jsonl: clean SIB-200 train records where label != topic (6 topics)
      - validation_id.jsonl: clean SIB-200 val records where label != topic
      - validation_ood.jsonl: OOD records from val where label == topic (1 topic)
      - test_id.jsonl: clean SIB-200 test records where label != topic
      - test_ood.jsonl: OOD records from test where label == topic (1 topic)
    """
    sib_path = Path(sib200_dir)
    out_base = Path(output_base_dir)

    train_recs = load_jsonl(sib_path / "train.jsonl")
    val_recs = load_jsonl(sib_path / "validation.jsonl")
    test_recs = load_jsonl(sib_path / "test.jsonl")

    # Discover the 7 SIB-200 canonical topics
    topics = sorted({r["labels"][0] for r in val_recs})
    if len(topics) != 7:
        raise ValueError(f"Expected 7 SIB-200 topics, found {len(topics)}: {topics}")

    fold_stats: dict[str, dict[str, int]] = {}

    for heldout_topic in topics:
        fold_dir = out_base / heldout_topic
        fold_dir.mkdir(parents=True, exist_ok=True)

        # ID splits (remaining 6 topics)
        train_id = [r for r in train_recs if r["labels"][0] != heldout_topic]
        val_id = [r for r in val_recs if r["labels"][0] != heldout_topic]
        test_id = [r for r in test_recs if r["labels"][0] != heldout_topic]

        # Near-OOD splits (held-out topic records formatted under OOD schema)
        val_ood_raw = [r for r in val_recs if r["labels"][0] == heldout_topic]
        test_ood_raw = [r for r in test_recs if r["labels"][0] == heldout_topic]

        val_ood: list[dict[str, Any]] = []
        for r in val_ood_raw:
            ood_rec = {
                "id": f"sib200:ood:{heldout_topic}:validation:{r['id']}",
                "text": r["text"],
                "language": r["language"],
                "source": "sib200",
                "license": r.get("license", "CC BY-SA 4.0"),
                "split": "validation",
                "is_ood": True,
                "ood_type": "near_ood",
                "ood_source": "sib200_heldout",
                "ood_reason": f"heldout_topic:{heldout_topic}",
                "source_id": r.get("source_id", r["id"]),
                "source_label": heldout_topic,
                "pair_id": r["pair_id"],
            }
            validate_ood_record(ood_rec)
            val_ood.append(ood_rec)

        test_ood: list[dict[str, Any]] = []
        for r in test_ood_raw:
            ood_rec = {
                "id": f"sib200:ood:{heldout_topic}:test:{r['id']}",
                "text": r["text"],
                "language": r["language"],
                "source": "sib200",
                "license": r.get("license", "CC BY-SA 4.0"),
                "split": "test",
                "is_ood": True,
                "ood_type": "near_ood",
                "ood_source": "sib200_heldout",
                "ood_reason": f"heldout_topic:{heldout_topic}",
                "source_id": r.get("source_id", r["id"]),
                "source_label": heldout_topic,
                "pair_id": r["pair_id"],
            }
            validate_ood_record(ood_rec)
            test_ood.append(ood_rec)

        # Strict validation checks
        # 1. Zero train leakage of heldout topic
        assert all(r["labels"][0] != heldout_topic for r in train_id)
        # 2. Train contains exactly 6 unique topics
        train_topics = {r["labels"][0] for r in train_id}
        assert len(train_topics) == 6
        assert heldout_topic not in train_topics
        # 3. Zero pair ID leakage between train ID and OOD evaluation
        train_pairs = {r["pair_id"] for r in train_id}
        ood_pairs = {r["pair_id"] for r in val_ood + test_ood}
        assert not (train_pairs & ood_pairs), f"Pair leakage in fold {heldout_topic}"
        # 4. EN/TR pairing preserved in ID and OOD splits
        assert len([r for r in val_ood if r["language"] == "en"]) == len([r for r in val_ood if r["language"] == "tr"])
        assert len([r for r in test_ood if r["language"] == "en"]) == len([r for r in test_ood if r["language"] == "tr"])

        # Validate ID records with normal schema validator
        for r in train_id:
            validate_record(r)
        for r in val_id:
            validate_record(r)
        for r in test_id:
            validate_record(r)

        save_jsonl(train_id, fold_dir / "train_id.jsonl")
        save_jsonl(val_id, fold_dir / "validation_id.jsonl")
        save_jsonl(val_ood, fold_dir / "validation_ood.jsonl")
        save_jsonl(test_id, fold_dir / "test_id.jsonl")
        save_jsonl(test_ood, fold_dir / "test_ood.jsonl")

        fold_stats[heldout_topic] = {
            "train_id": len(train_id),
            "validation_id": len(val_id),
            "validation_ood": len(val_ood),
            "test_id": len(test_id),
            "test_ood": len(test_ood),
            "total_ood": len(val_ood) + len(test_ood),
        }

    return fold_stats


def download_and_extract_massive(
    raw_dir: str | Path = "data/raw/massive",
) -> tuple[Path, Path]:
    """Download and extract official Amazon MASSIVE en-US and tr-TR files if not present."""
    target_dir = Path(raw_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    en_file = target_dir / "en-US.jsonl"
    tr_file = target_dir / "tr-TR.jsonl"

    if en_file.exists() and tr_file.exists():
        return en_file, tr_file

    print(f"Downloading MASSIVE dataset archive from: {MASSIVE_TAR_URL}")
    with urllib.request.urlopen(MASSIVE_TAR_URL) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                if member.name == "1.1/data/en-US.jsonl":
                    f = tar.extractfile(member)
                    if f:
                        en_file.write_bytes(f.read())
                elif member.name == "1.1/data/tr-TR.jsonl":
                    f = tar.extractfile(member)
                    if f:
                        tr_file.write_bytes(f.read())

    return en_file, tr_file


def load_massive_paired_records(
    en_path: str | Path,
    tr_path: str | Path,
    allowed_scenarios: set[str] = MASSIVE_ALLOWED_SCENARIOS,
) -> dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]:
    """Load, filter, and pair MASSIVE utterances by ID across en-US and tr-TR."""
    en_records = load_jsonl(en_path)
    tr_records = load_jsonl(tr_path)

    # Index Turkish records by ID
    tr_by_id = {r["id"]: r for r in tr_records}

    paired_by_partition: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {
        "dev": [],
        "test": [],
        "train": [],
    }

    for en_rec in en_records:
        scenario = en_rec.get("scenario")
        if scenario not in allowed_scenarios:
            continue

        item_id = en_rec["id"]
        if item_id not in tr_by_id:
            continue
        tr_rec = tr_by_id[item_id]

        # Verify scenario, intent, and partition match
        if (
            tr_rec.get("scenario") != scenario
            or tr_rec.get("intent") != en_rec.get("intent")
            or tr_rec.get("partition") != en_rec.get("partition")
        ):
            continue

        partition = en_rec.get("partition", "test")
        if partition in paired_by_partition:
            paired_by_partition[partition].append((en_rec, tr_rec))

    # Sort deterministically by integer ID
    for partition in paired_by_partition:
        paired_by_partition[partition].sort(key=lambda pair: int(pair[0]["id"]))

    return paired_by_partition


def generate_far_ood_benchmarks(
    raw_dir: str | Path = "data/raw/massive",
    output_base_dir: str | Path = "data/ood/far/massive",
    val_sample_size: int = 250,
    test_sample_size: int = 500,
    seed: int = 42,
) -> dict[str, int]:
    """Generate deterministic, balanced EN/TR Far-OOD benchmarks from MASSIVE."""
    en_path, tr_path = download_and_extract_massive(raw_dir)
    paired_by_partition = load_massive_paired_records(en_path, tr_path)

    out_dir = Path(output_base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    # Sample matched pairs for validation and test
    dev_pairs = list(paired_by_partition["dev"])
    test_pairs = list(paired_by_partition["test"])

    if len(dev_pairs) < val_sample_size:
        raise ValueError(
            f"Insufficient matched dev pairs: requested {val_sample_size}, but only {len(dev_pairs)} available."
        )
    if len(test_pairs) < test_sample_size:
        raise ValueError(
            f"Insufficient matched test pairs: requested {test_sample_size}, but only {len(test_pairs)} available."
        )

    rng.shuffle(dev_pairs)
    rng.shuffle(test_pairs)

    selected_val_pairs = dev_pairs[:val_sample_size]
    selected_test_pairs = test_pairs[:test_sample_size]

    # Sort by ID for deterministic, stable file order
    selected_val_pairs.sort(key=lambda p: int(p[0]["id"]))
    selected_test_pairs.sort(key=lambda p: int(p[0]["id"]))

    results: dict[str, int] = {}
    expected_pair_counts = {"validation": val_sample_size, "test": test_sample_size}

    for split_name, pairs in (("validation", selected_val_pairs), ("test", selected_test_pairs)):
        expected_pairs = expected_pair_counts[split_name]
        ood_records: list[dict[str, Any]] = []
        for en_rec, tr_rec in pairs:
            item_id = en_rec["id"]
            scenario = en_rec["scenario"]
            intent = en_rec["intent"]
            pair_id = f"massive:{item_id}"

            # English OOD record
            en_ood = {
                "id": f"massive:{split_name}:{item_id}:en",
                "text": en_rec["utt"].strip(),
                "language": "en",
                "source": "massive",
                "license": "CC BY 4.0",
                "split": split_name,
                "is_ood": True,
                "ood_type": "far_ood",
                "ood_source": "massive",
                "ood_reason": f"smart_assistant_command:{scenario}",
                "source_id": item_id,
                "source_label": f"{scenario}:{intent}",
                "pair_id": pair_id,
            }
            validate_ood_record(en_ood)
            ood_records.append(en_ood)

            # Turkish OOD record
            tr_ood = {
                "id": f"massive:{split_name}:{item_id}:tr",
                "text": tr_rec["utt"].strip(),
                "language": "tr",
                "source": "massive",
                "license": "CC BY 4.0",
                "split": split_name,
                "is_ood": True,
                "ood_type": "far_ood",
                "ood_source": "massive",
                "ood_reason": f"smart_assistant_command:{scenario}",
                "source_id": item_id,
                "source_label": f"{scenario}:{intent}",
                "pair_id": pair_id,
            }
            validate_ood_record(tr_ood)
            ood_records.append(tr_ood)

        # Verify uniqueness
        ids = [r["id"] for r in ood_records]
        assert len(ids) == len(set(ids)), f"Duplicate IDs in Far-OOD {split_name}"

        # Verify exact requested pair counts and exact 1:1 English/Turkish pair_id matching
        assert len(pairs) == expected_pairs, (
            f"Expected {expected_pairs} pairs for Far-OOD {split_name}, got {len(pairs)}"
        )
        assert len(ood_records) == 2 * expected_pairs, (
            f"Expected {2 * expected_pairs} records for Far-OOD {split_name}, got {len(ood_records)}"
        )
        pair_counts = Counter(r["pair_id"] for r in ood_records)
        assert len(pair_counts) == expected_pairs, (
            f"Expected {expected_pairs} unique pair_ids, got {len(pair_counts)}"
        )
        assert all(cnt == 2 for cnt in pair_counts.values()), (
            f"Every pair_id must appear exactly twice in Far-OOD {split_name}"
        )
        pair_langs: dict[str, set[str]] = {}
        for r in ood_records:
            pair_langs.setdefault(r["pair_id"], set()).add(r["language"])
        assert all(langs == {"en", "tr"} for langs in pair_langs.values()), (
            f"Every pair_id must have exactly one 'en' and one 'tr' record in Far-OOD {split_name}"
        )

        out_file = out_dir / f"{split_name}_ood.jsonl"
        save_jsonl(ood_records, out_file)
        results[split_name] = len(ood_records)

    return results
