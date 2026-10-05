"""Out-of-Distribution (OOD) benchmark generation for NALTRA.

Uses SIB-200 Leave-One-Topic-Out to create deterministic Near-OOD folds
without introducing any additional dataset.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from naltra.data.loader import load_jsonl, save_jsonl
from naltra.data.preprocessing import validate_record

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

    if record["ood_type"] != "near_ood":
        raise ValueError(
            f"OOD record '{record['id']}' has invalid ood_type '{record['ood_type']}'. "
            "Must be 'near_ood'."
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
