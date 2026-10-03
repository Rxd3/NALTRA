"""Out-of-Distribution (OOD) benchmark generation for NALTRA.

Supports two separate, complementary OOD evaluation settings:
1. Near-OOD: SIB-200 Leave-One-Topic-Out (7 deterministic topic folds).
2. Far-OOD: Amazon MASSIVE dataset (voice-assistant commands filtered to non-news domains).
"""

from __future__ import annotations

import random
import tarfile
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from naltra.data.loader import load_jsonl, save_jsonl
from naltra.data.preprocessing import validate_record

MASSIVE_TAR_URL = (
    "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
)

# Explicit allowlist of voice-assistant command scenarios that do not overlap SIB-200 news topics
MASSIVE_ALLOWED_SCENARIOS: set[str] = {
    "alarm",  # alarm_set, alarm_query, alarm_remove
    "datetime",  # datetime_query, datetime_convert
    "calendar",  # calendar_set, calendar_query, calendar_remove
    "lists",  # lists_createoradd, lists_query, lists_remove
    "iot",  # iot_hue_lighton, iot_hue_lightoff, iot_coffee, iot_wemo_on, etc.
    "audio",  # audio_volume_up, audio_volume_down, audio_volume_mute, audio_volume_other
    "takeaway",  # takeaway_order, takeaway_query
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
    """Validate that an OOD record complies strictly with the NALTRA OOD schema."""
    if not isinstance(record, dict):
        raise ValueError("OOD record must be a dictionary.")

    for field in REQUIRED_OOD_FIELDS:
        if field not in record:
            raise ValueError(
                f"OOD record '{record.get('id', 'unknown')}' is missing required field '{field}'."
            )

    string_fields = (
        "id",
        "text",
        "source",
        "license",
        "ood_source",
        "ood_reason",
        "source_label",
        "pair_id",
    )
    for field in string_fields:
        val = record[field]
        if not isinstance(val, str) or not val.strip():
            raise ValueError(
                f"OOD record '{record.get('id', 'unknown')}' field '{field}' "
                f"must be a non-empty string, got {val!r} ({type(val).__name__})."
            )

    src_id = record["source_id"]
    if isinstance(src_id, bool) or not isinstance(src_id, (str, int)) or not str(src_id).strip():
        raise ValueError(
            f"OOD record '{record['id']}' field 'source_id' must be a valid "
            f"non-empty scalar identifier (string or integer, not bool), "
            f"got {src_id!r} ({type(src_id).__name__})."
        )

    if record["language"] not in ("en", "tr"):
        raise ValueError(
            f"OOD record '{record['id']}' has invalid language '{record['language']}'. "
            "Must be 'en' or 'tr'."
        )

    if record["split"] not in ("validation", "test"):
        raise ValueError(
            f"OOD record '{record['id']}' has invalid split '{record['split']}'. "
            "Must be 'validation' or 'test'."
        )

    if record["is_ood"] is not True:
        raise ValueError(f"OOD record '{record['id']}' must have is_ood=True.")

    if record["ood_type"] not in ("near_ood", "far_ood"):
        raise ValueError(
            f"OOD record '{record['id']}' has invalid ood_type '{record['ood_type']}'. "
            "Must be 'near_ood' or 'far_ood'."
        )

    # Critical requirement: OOD records must NOT contain non-empty canonical labels
    if "labels" in record and record["labels"]:
        raise ValueError(
            f"OOD record '{record['id']}' must not be assigned canonical taxonomy labels. "
            f"Got: {record['labels']}"
        )


def validate_sib200_source_data(
    train_recs: Sequence[dict[str, Any]],
    val_recs: Sequence[dict[str, Any]],
    test_recs: Sequence[dict[str, Any]],
) -> None:
    """Strictly validate source SIB-200 data before Near-OOD fold generation."""
    all_recs = list(train_recs) + list(val_recs) + list(test_recs)

    # 1. Every record passes validate_record()
    for r in all_recs:
        validate_record(r)

    # 2. No duplicate record IDs across all records
    seen_ids: set[str] = set()
    for r in all_recs:
        rec_id = r["id"]
        if rec_id in seen_ids:
            raise ValueError(f"Duplicate SIB-200 record ID detected: '{rec_id}'")
        seen_ids.add(rec_id)

    # 3. Disjoint record IDs across splits
    train_ids = {r["id"] for r in train_recs}
    val_ids = {r["id"] for r in val_recs}
    test_ids = {r["id"] for r in test_recs}
    if train_ids & val_ids:
        raise ValueError("SIB-200 record IDs overlap between train and validation splits.")
    if train_ids & test_ids:
        raise ValueError("SIB-200 record IDs overlap between train and test splits.")
    if val_ids & test_ids:
        raise ValueError("SIB-200 record IDs overlap between validation and test splits.")

    # 4. Disjoint pair IDs across splits
    train_pairs = {r.get("pair_id") for r in train_recs if r.get("pair_id")}
    val_pairs = {r.get("pair_id") for r in val_recs if r.get("pair_id")}
    test_pairs = {r.get("pair_id") for r in test_recs if r.get("pair_id")}
    if train_pairs & val_pairs:
        raise ValueError("SIB-200 pair IDs overlap between train and validation splits.")
    if train_pairs & test_pairs:
        raise ValueError("SIB-200 pair IDs overlap between train and test splits.")
    if val_pairs & test_pairs:
        raise ValueError("SIB-200 pair IDs overlap between validation and test splits.")

    # 5. Every record has a valid non-empty pair_id
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in all_recs:
        p_id = r.get("pair_id")
        if not p_id or not isinstance(p_id, str) or not p_id.strip():
            raise ValueError(f"SIB-200 record '{r['id']}' is missing a non-empty pair_id.")
        pairs[p_id].append(r)

    # 6. Every pair contains exactly one EN and one TR record with matching labels and split
    for p_id, recs in pairs.items():
        if len(recs) != 2:
            raise ValueError(
                f"SIB-200 pair '{p_id}' has {len(recs)} records; expected exactly 2 (1 EN, 1 TR)."
            )
        en_list = [r for r in recs if r.get("language") == "en"]
        tr_list = [r for r in recs if r.get("language") == "tr"]
        if len(en_list) != 1 or len(tr_list) != 1:
            raise ValueError(
                f"SIB-200 pair '{p_id}' must have exactly one 'en' and one 'tr' record, "
                f"got {len(en_list)} en and {len(tr_list)} tr."
            )
        en_rec, tr_rec = en_list[0], tr_list[0]
        if en_rec.get("labels") != tr_rec.get("labels"):
            raise ValueError(
                f"SIB-200 pair '{p_id}' has mismatched labels: "
                f"EN has {en_rec.get('labels')}, TR has {tr_rec.get('labels')}."
            )
        if en_rec.get("split") != tr_rec.get("split"):
            raise ValueError(
                f"SIB-200 pair '{p_id}' has mismatched splits: "
                f"EN has '{en_rec.get('split')}', TR has '{tr_rec.get('split')}'."
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
    from naltra.data.manifest import create_manifest

    sib_path = Path(sib200_dir)
    out_base = Path(output_base_dir)

    # Preflight check: required SIB-200 clean files must exist
    missing_files = [
        str(sib_path / f"{s}.jsonl")
        for s in ("train", "validation", "test")
        if not (sib_path / f"{s}.jsonl").exists()
    ]
    if missing_files:
        files_str = "\n  - ".join(missing_files)
        raise FileNotFoundError(
            "Cannot generate Near-OOD benchmarks because required SIB-200 input files "
            f"are missing:\n"
            f"  - {files_str}\n"
            f"Please prepare SIB-200 first:\n"
            f"  python scripts/prepare_data.py --dataset sib200"
        )

    train_recs = load_jsonl(sib_path / "train.jsonl")
    val_recs = load_jsonl(sib_path / "validation.jsonl")
    test_recs = load_jsonl(sib_path / "test.jsonl")

    # Strictly validate source SIB-200 data before generating folds
    validate_sib200_source_data(train_recs, val_recs, test_recs)

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
        if any(r["labels"][0] == heldout_topic for r in train_id):
            raise RuntimeError(f"Held-out topic '{heldout_topic}' leaked into ID training set!")
        # 2. Train contains exactly 6 unique topics
        train_topics = {r["labels"][0] for r in train_id}
        if len(train_topics) != 6 or heldout_topic in train_topics:
            raise RuntimeError(
                f"ID training set must contain exactly 6 topics excluding '{heldout_topic}', "
                f"got {train_topics}"
            )
        # 3. Zero pair ID leakage between train ID and OOD evaluation
        train_pairs = {r["pair_id"] for r in train_id}
        ood_pairs = {r["pair_id"] for r in val_ood + test_ood}
        if train_pairs & ood_pairs:
            raise RuntimeError(
                f"Pair ID leakage detected in Near-OOD fold '{heldout_topic}': "
                f"{train_pairs & ood_pairs}"
            )
        # 4. EN/TR pairing preserved in ID and OOD splits
        en_val_count = len([r for r in val_ood if r["language"] == "en"])
        tr_val_count = len([r for r in val_ood if r["language"] == "tr"])
        if en_val_count != tr_val_count:
            raise RuntimeError(
                f"EN/TR count mismatch in validation_ood: {en_val_count} en vs {tr_val_count} tr"
            )

        en_test_count = len([r for r in test_ood if r["language"] == "en"])
        tr_test_count = len([r for r in test_ood if r["language"] == "tr"])
        if en_test_count != tr_test_count:
            raise RuntimeError(
                f"EN/TR count mismatch in test_ood: {en_test_count} en vs {tr_test_count} tr"
            )

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

        # Create reproducibility manifest for this fold
        create_manifest(
            benchmark_name=f"near_ood_sib200_{heldout_topic}",
            output_dir=fold_dir,
            generation_parameters={
                "heldout_topic": heldout_topic,
                "source": "sib200",
            },
            source_metadata={
                "sib200_dir": str(sib_path),
            },
        )

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
            f"Insufficient matched dev pairs: requested {val_sample_size}, "
            f"but only {len(dev_pairs)} available."
        )
    if len(test_pairs) < test_sample_size:
        raise ValueError(
            f"Insufficient matched test pairs: requested {test_sample_size}, "
            f"but only {len(test_pairs)} available."
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
        assert (
            len(pairs) == expected_pairs
        ), f"Expected {expected_pairs} pairs for Far-OOD {split_name}, got {len(pairs)}"
        assert (
            len(ood_records) == 2 * expected_pairs
        ), f"Expected {2 * expected_pairs} records for Far-OOD {split_name}, got {len(ood_records)}"
        pair_counts = Counter(r["pair_id"] for r in ood_records)
        assert (
            len(pair_counts) == expected_pairs
        ), f"Expected {expected_pairs} unique pair_ids, got {len(pair_counts)}"
        assert all(
            cnt == 2 for cnt in pair_counts.values()
        ), f"Every pair_id must appear exactly twice in Far-OOD {split_name}"
        pair_langs: dict[str, set[str]] = {}
        for r in ood_records:
            pair_langs.setdefault(r["pair_id"], set()).add(r["language"])
        assert all(
            langs == {"en", "tr"} for langs in pair_langs.values()
        ), f"Every pair_id must have exactly one 'en' and one 'tr' record in Far-OOD {split_name}"

        out_file = out_dir / f"{split_name}_ood.jsonl"
        save_jsonl(ood_records, out_file)
        results[split_name] = len(ood_records)

    # Generate reproducibility manifest for Far-OOD
    from naltra.data.manifest import compute_file_sha256, create_manifest

    create_manifest(
        benchmark_name="far_ood_massive",
        output_dir=out_dir,
        generation_parameters={
            "val_sample_size": val_sample_size,
            "test_sample_size": test_sample_size,
            "seed": seed,
            "allowed_scenarios": sorted(MASSIVE_ALLOWED_SCENARIOS),
        },
        source_metadata={
            "source": "Amazon MASSIVE 1.1",
            "en_sha256": compute_file_sha256(en_path),
            "tr_sha256": compute_file_sha256(tr_path),
        },
    )

    return results
