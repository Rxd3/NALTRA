"""Shared dataset preprocessing, schema validation, and preparation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from naltra.data.loader import save_jsonl
from naltra.data.manifest import (
    REPO_ROOT,
    compute_file_md5,
    create_manifest,
    get_source_config,
)
from naltra.pipeline.preprocessing import normalize_text

REQUIRED_RECORD_FIELDS = (
    "id",
    "text",
    "labels",
    "language",
    "source",
    "license",
    "split",
)
VALID_LANGUAGES = {"en", "tr", "en-tr"}
VALID_SPLITS = {"train", "validation", "test"}


def preprocess_record_text(text: str) -> str:
    """Apply the shared, language-preserving text normalization."""
    return normalize_text(text)


def compute_content_fingerprint(text: str) -> str:
    """Compute a deterministic SHA-256 fingerprint over normalized text representation.

    Uses preprocess_record_text(text) to ensure the exact same normalization used by
    processed benchmark records, providing a platform-independent hash for detecting
    exact content duplicates without replacing unique record IDs.
    """
    import hashlib

    normalized = preprocess_record_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_label_map(path: str | Path = "taxonomy/label_map.json") -> dict[str, str]:
    """Load source-to-canonical label aliases from label_map.json."""
    map_path = Path(path)
    with map_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    aliases = data.get("aliases")
    if not isinstance(aliases, dict):
        raise ValueError(f"Expected 'aliases' dict in {path}")
    return aliases


def load_canonical_label_ids(path: str | Path = "taxonomy/taxonomy.json") -> set[str]:
    """Load valid canonical label identifiers from taxonomy.json."""
    tax_path = Path(path)
    with tax_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    labels = data.get("labels", [])
    return {
        label["id"]
        for label in labels
        if isinstance(label, dict) and isinstance(label.get("id"), str)
    }


def map_label(source_label: str, label_map: dict[str, str]) -> str:
    """Map a single source label to its canonical ID, failing loudly if unmapped."""
    if source_label in label_map:
        return label_map[source_label]
    raise KeyError(
        f"Unknown source label: '{source_label}'. "
        "Please add it to taxonomy/label_map.json before preprocessing."
    )


def map_labels(source_labels: list[str] | str, label_map: dict[str, str]) -> list[str]:
    """Map source labels to unique canonical IDs, preserving order."""
    if isinstance(source_labels, str):
        source_labels = [source_labels]
    canonical = []
    seen = set()
    for label in source_labels:
        canonical_id = map_label(label, label_map)
        if canonical_id not in seen:
            seen.add(canonical_id)
            canonical.append(canonical_id)
    return canonical


def validate_record(
    record: dict[str, Any],
    allowed_labels: set[str] | None = None,
) -> None:
    """Validate a normalized record against the NALTRA schema."""
    if not isinstance(record, dict):
        raise ValueError("Record must be a dictionary.")

    for field in REQUIRED_RECORD_FIELDS:
        if field not in record:
            raise ValueError(
                f"Record '{record.get('id', '<unknown>')}' missing required field: '{field}'"
            )

    if not isinstance(record["id"], str) or not record["id"].strip():
        raise ValueError("Record 'id' must be a non-empty string.")

    if not isinstance(record["text"], str) or not record["text"].strip():
        raise ValueError(f"Record '{record['id']}' has empty or non-string 'text'.")

    labels = record["labels"]
    if not isinstance(labels, list) or len(labels) == 0:
        raise ValueError(f"Record '{record['id']}' must have a non-empty list of labels.")

    for label in labels:
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Record '{record['id']}' has invalid label item: {label!r}")
        if allowed_labels is not None and label not in allowed_labels:
            raise ValueError(
                f"Record '{record['id']}' contains label '{label}' not in canonical taxonomy."
            )

    if record["language"] not in VALID_LANGUAGES:
        raise ValueError(
            f"Record '{record['id']}' has invalid language '{record['language']}'. "
            f"Allowed: {sorted(VALID_LANGUAGES)}"
        )

    if not isinstance(record["source"], str) or not record["source"].strip():
        raise ValueError(f"Record '{record['id']}' has invalid 'source'.")

    if not isinstance(record["license"], str) or not record["license"].strip():
        raise ValueError(f"Record '{record['id']}' has invalid 'license'.")

    if record["split"] not in VALID_SPLITS:
        raise ValueError(
            f"Record '{record['id']}' has invalid split '{record['split']}'. "
            f"Allowed: {sorted(VALID_SPLITS)}"
        )


def process_sib200(
    output_dir: str | Path = "data/processed/sib200",
    taxonomy_path: str | Path = "taxonomy/taxonomy.json",
    label_map_path: str | Path = "taxonomy/label_map.json",
) -> dict[str, list[dict[str, Any]]]:
    """Download, preprocess, validate, and persist SIB-200 English and Turkish subsets."""
    from datasets import load_dataset

    label_map = load_label_map(label_map_path)
    allowed_labels = load_canonical_label_ids(taxonomy_path)

    configs = {
        "en": "eng_Latn",
        "tr": "tur_Latn",
    }
    splits = ["train", "validation", "test"]

    print("Loading SIB-200 English and Turkish subsets...")
    source = get_source_config("sib200")
    raw_datasets = {
        lang: load_dataset(source["dataset_id"], cfg, revision=source["revision"])
        for lang, cfg in configs.items()
    }

    processed_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in splits}

    for split_name in splits:
        records: list[dict[str, Any]] = []
        for lang in ("en", "tr"):
            split_data = raw_datasets[lang][split_name]
            for row in split_data:
                index_id = row["index_id"]
                raw_category = row["category"]
                canonical_labels = map_labels([raw_category], label_map)

                record: dict[str, Any] = {
                    "id": f"sib200:{lang}:{index_id}",
                    "text": preprocess_record_text(row["text"]),
                    "labels": canonical_labels,
                    "language": lang,
                    "source": "sib200",
                    "license": "CC BY-SA 4.0",
                    "split": split_name,
                    "pair_id": f"sib200:{index_id}",
                    "source_id": index_id,
                    "source_labels": [raw_category],
                }
                validate_record(record, allowed_labels=allowed_labels)
                records.append(record)

        # Check unique IDs within split
        ids = [r["id"] for r in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate IDs found in SIB-200 {split_name} split.")

        out_path = Path(output_dir) / f"{split_name}.jsonl"
        save_jsonl(records, out_path)
        processed_by_split[split_name] = records
        half = len(records) // 2
        print(f"Saved {len(records)} records ({half} en, {half} tr) -> {out_path}")

    create_manifest(
        "clean_sib200",
        output_dir,
        generation_parameters={"configs": configs, "splits": splits},
        source_metadata={**source, "license": "CC BY-SA 4.0"},
        taxonomy_dir=Path(taxonomy_path).parent,
        input_files=[label_map_path, REPO_ROOT / "configs/data.yaml"],
    )
    return processed_by_split


def process_multifin(
    output_dir: str | Path = "data/processed/multifin",
    taxonomy_path: str | Path = "taxonomy/taxonomy.json",
    label_map_path: str | Path = "taxonomy/label_map.json",
) -> dict[str, list[dict[str, Any]]]:
    """Download, preprocess, validate, and persist MultiFin English and Turkish subsets."""
    from datasets import load_dataset

    label_map = load_label_map(label_map_path)
    allowed_labels = load_canonical_label_ids(taxonomy_path)

    target_languages = {"English": "en", "Turkish": "tr"}
    splits = ["train", "validation", "test"]

    print("Loading MultiFin all_languages_lowlevel dataset...")
    source = get_source_config("multifin")
    raw_dataset = load_dataset(
        source["dataset_id"], "all_languages_lowlevel", revision=source["revision"]
    )

    processed_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in splits}

    for split_name in splits:
        records: list[dict[str, Any]] = []
        split_data = raw_dataset[split_name]
        for row in split_data:
            raw_lang = row.get("lang")
            if raw_lang not in target_languages:
                continue

            lang = target_languages[raw_lang]
            raw_id = row["id"]
            raw_labels = row["labels"]
            canonical_labels = map_labels(raw_labels, label_map)

            record: dict[str, Any] = {
                "id": f"multifin:{raw_id}",
                "text": preprocess_record_text(row["text"]),
                "labels": canonical_labels,
                "language": lang,
                "source": "multifin",
                "license": "CC BY-NC 4.0",
                "split": split_name,
                "source_id": raw_id,
                "source_labels": list(raw_labels),
            }
            validate_record(record, allowed_labels=allowed_labels)
            records.append(record)

        # Check unique IDs within split
        ids = [r["id"] for r in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate IDs found in MultiFin {split_name} split.")

        out_path = Path(output_dir) / f"{split_name}.jsonl"
        save_jsonl(records, out_path)
        processed_by_split[split_name] = records
        en_count = sum(1 for r in records if r["language"] == "en")
        tr_count = sum(1 for r in records if r["language"] == "tr")
        print(f"Saved {len(records)} records ({en_count} en, {tr_count} tr) -> {out_path}")

    create_manifest(
        "clean_multifin_official",
        output_dir,
        generation_parameters={
            "config": "all_languages_lowlevel",
            "splits": splits,
            "languages": target_languages,
        },
        source_metadata={**source, "license": "CC BY-NC 4.0"},
        taxonomy_dir=Path(taxonomy_path).parent,
        input_files=[label_map_path, REPO_ROOT / "configs/data.yaml"],
    )
    return processed_by_split


def generate_multifin_leakage_free_track(
    official_multifin_dir: str | Path = "data/processed/multifin",
    output_dir: str | Path = "data/splits/multifin/leakage_free",
    taxonomy_path: str | Path = "taxonomy/taxonomy.json",
) -> dict[str, Any]:
    """Preserve official training and retained test records; remove train contamination
    from evaluation and test-content overlap from validation before model selection.
    """
    import copy

    from naltra.data.loader import load_jsonl, save_jsonl

    allowed_labels = load_canonical_label_ids(taxonomy_path)
    in_dir = Path(official_multifin_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_file = in_dir / "train.jsonl"
    val_file = in_dir / "validation.jsonl"
    test_file = in_dir / "test.jsonl"

    if not train_file.exists() or not val_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            f"Official MultiFin files missing in {in_dir}. Run prepare_multifin first."
        )

    official_train = load_jsonl(train_file)
    official_val = load_jsonl(val_file)
    official_test = load_jsonl(test_file)

    train_fps = {compute_content_fingerprint(r["text"]) for r in official_train}

    clean_train = [copy.deepcopy(r) for r in official_train]
    for r in clean_train:
        r["original_split"] = "train"
        r["split"] = "train"
        validate_record(r, allowed_labels=allowed_labels)

    clean_val = []
    val_removed = []
    for r in official_val:
        if compute_content_fingerprint(r["text"]) in train_fps:
            val_removed.append(r)
        else:
            rec = copy.deepcopy(r)
            rec["original_split"] = "validation"
            rec["split"] = "validation"
            validate_record(rec, allowed_labels=allowed_labels)
            clean_val.append(rec)

    clean_test = []
    test_removed = []
    for r in official_test:
        if compute_content_fingerprint(r["text"]) in train_fps:
            test_removed.append(r)
        else:
            rec = copy.deepcopy(r)
            rec["original_split"] = "test"
            rec["split"] = "test"
            validate_record(rec, allowed_labels=allowed_labels)
            clean_test.append(rec)

    test_fps = {compute_content_fingerprint(r["text"]) for r in clean_test}
    val_test_removed = [r for r in clean_val if compute_content_fingerprint(r["text"]) in test_fps]
    clean_val = [r for r in clean_val if compute_content_fingerprint(r["text"]) not in test_fps]
    val_fps = {compute_content_fingerprint(r["text"]) for r in clean_val}
    val_test_overlap = val_fps & test_fps

    save_jsonl(clean_train, out_dir / "train.jsonl")
    save_jsonl(clean_val, out_dir / "validation.jsonl")
    save_jsonl(clean_test, out_dir / "test.jsonl")

    stats = {
        "train_count": len(clean_train),
        "official_val_count": len(official_val),
        "clean_val_count": len(clean_val),
        "val_removed_count": len(val_removed) + len(val_test_removed),
        "val_train_removed_count": len(val_removed),
        "val_test_removed_count": len(val_test_removed),
        "official_test_count": len(official_test),
        "clean_test_count": len(clean_test),
        "test_removed_count": len(test_removed),
        "val_test_overlap_count": len(val_test_overlap),
    }

    create_manifest(
        benchmark_name="multifin_leakage_free",
        output_dir=out_dir,
        generation_parameters={
            "official_multifin_dir": str(in_dir),
            "removed_val_leaks": len(val_removed),
            "removed_test_leaks": len(test_removed),
            "removed_val_test_overlap": len(val_test_removed),
            "overlap_policy": "preserve_test_remove_from_validation",
            "val_test_overlap_count": len(val_test_overlap),
        },
        source_metadata={
            "official_train_count": len(official_train),
            "official_val_count": len(official_val),
            "official_test_count": len(official_test),
        },
        taxonomy_dir=Path(taxonomy_path).parent,
        input_files=[train_file, val_file, test_file],
    )

    print(
        f"MultiFin Leakage-Free Track Generated -> {out_dir}:\n"
        f"  Train: {stats['train_count']} (reference)\n"
        f"  Validation: {stats['clean_val_count']} "
        f"(removed {len(val_removed)} train leaks, {len(val_test_removed)} test overlaps)\n"
        f"  Test: {stats['clean_test_count']} "
        f"(removed {stats['test_removed_count']} leaking from train)\n"
        f"  Val/Test Overlap Fingerprints: {stats['val_test_overlap_count']}"
    )

    return stats


def process_mn_ds(
    csv_path: str | Path = "data/raw/mn_ds/MN-DS-news-classification.csv",
    output_dir: str | Path = "data/processed/mn_ds",
    taxonomy_path: str | Path = "taxonomy/taxonomy.json",
    label_map_path: str | Path = "taxonomy/label_map.json",
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Load, group, preprocess, deterministically split, validate, and persist MN-DS."""
    import pandas as pd

    from naltra.data.splits import grouped_multilabel_stratified_split

    label_map = load_label_map(label_map_path)
    allowed_labels = load_canonical_label_ids(taxonomy_path)

    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"MN-DS CSV not found at: {csv_file}")
    source = get_source_config("mn_ds")
    if compute_file_md5(csv_file) != source["md5"]:
        raise ValueError("MN-DS source checksum mismatch; download the verified Zenodo file.")

    print(f"Loading MN-DS raw data from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} annotation rows.")

    # Group by article 'id' to prevent train/test leakage
    articles: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        article_id = str(row["id"]).strip()
        raw_l2 = str(row["category_level_2"]).strip()

        if article_id not in articles:
            title = str(row["title"]).strip() if pd.notna(row["title"]) else ""
            content = str(row["content"]).strip() if pd.notna(row["content"]) else ""

            # Combine title and content cleanly
            if title and content:
                if title.endswith((".", "!", "?")):
                    raw_text = f"{title} {content}"
                else:
                    raw_text = f"{title}. {content}"
            elif content:
                raw_text = content
            else:
                raw_text = title

            articles[article_id] = {
                "id": f"mn_ds:{article_id}",
                "text": preprocess_record_text(raw_text),
                "source_labels": [],
                "language": "en",
                "source": "mn_ds",
                "license": "CC BY 4.0",
                "source_id": article_id,
            }

        if raw_l2 and raw_l2 not in articles[article_id]["source_labels"]:
            articles[article_id]["source_labels"].append(raw_l2)

    print(f"Merged into {len(articles)} unique articles.")

    # Map labels to canonical IDs
    article_records: list[dict[str, Any]] = []
    for rec in articles.values():
        rec["labels"] = map_labels(rec["source_labels"], label_map)
        article_records.append(rec)

    # Perform deterministic multi-label stratified split with content-fingerprint grouping
    print(
        f"Grouping and splitting {len(article_records)} articles using content-grouped "
        f"iterative stratification (seed={seed})..."
    )
    train_records, val_records, test_records = grouped_multilabel_stratified_split(
        article_records,
        group_key=lambda r: compute_content_fingerprint(r["text"]),
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    splits = {
        "train": train_records,
        "validation": val_records,
        "test": test_records,
    }

    # Verify zero ID leakage across splits
    train_ids = {r["id"] for r in train_records}
    val_ids = {r["id"] for r in val_records}
    test_ids = {r["id"] for r in test_records}

    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("Data leakage detected: article IDs overlap across splits!")

    # Verify zero content-fingerprint leakage across splits
    train_fps = {compute_content_fingerprint(r["text"]) for r in train_records}
    val_fps = {compute_content_fingerprint(r["text"]) for r in val_records}
    test_fps = {compute_content_fingerprint(r["text"]) for r in test_records}

    if train_fps & val_fps or train_fps & test_fps or val_fps & test_fps:
        raise ValueError("Content fingerprint leakage detected across MN-DS splits!")

    # Validate and save each split
    out_dir = Path(output_dir)
    for split_name, split_data in splits.items():
        for record in split_data:
            record["split"] = split_name
            validate_record(record, allowed_labels=allowed_labels)

        out_path = out_dir / f"{split_name}.jsonl"
        save_jsonl(split_data, out_path)
        print(f"Saved {len(split_data)} records -> {out_path}")

    create_manifest(
        "clean_mn_ds",
        out_dir,
        generation_parameters={
            "seed": seed,
            "train_ratio": train_ratio,
            "validation_ratio": validation_ratio,
            "test_ratio": test_ratio,
            "split_method": "content_grouped_multilabel_stratification",
        },
        source_metadata={**source, "license": "CC BY 4.0"},
        taxonomy_dir=Path(taxonomy_path).parent,
        input_files=[csv_file, label_map_path],
    )
    return splits
