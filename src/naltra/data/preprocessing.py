"""Shared dataset preprocessing, schema validation, and preparation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from naltra.data.loader import save_jsonl
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
            raise ValueError(f"Record '{record.get('id', '<unknown>')}' missing required field: '{field}'")

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
    raw_datasets = {
        lang: load_dataset("Davlan/sib200", cfg)
        for lang, cfg in configs.items()
    }

    processed_by_split: dict[str, list[dict[str, Any]]] = {
        split: [] for split in splits
    }

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
        print(f"Saved {len(records)} records ({len(records)//2} en, {len(records)//2} tr) -> {out_path}")

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
    raw_dataset = load_dataset("awinml/MultiFin", "all_languages_lowlevel")

    processed_by_split: dict[str, list[dict[str, Any]]] = {
        split: [] for split in splits
    }

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

    return processed_by_split
