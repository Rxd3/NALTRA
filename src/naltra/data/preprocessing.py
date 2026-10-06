"""Shared dataset preprocessing, schema validation, and preparation helpers."""

from __future__ import annotations

import html
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from naltra.data.loader import save_jsonl
from naltra.data.manifest import create_manifest
from naltra.data.splits import grouped_multilabel_stratified_split
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
    """Apply the shared, language-preserving scientific text normalization.

    Removes HTML markup and unescapes HTML entities, normalizes consecutive whitespace
    and line breaks, while strictly preserving scientific notations, mathematical formulas,
    chemical formulas, punctuation, numbers, technical terminology, and casing.
    """
    if not text:
        return ""
    # Strip HTML tags
    cleaned = re.sub(r"<[^>]+>", " ", text)
    # Unescape HTML entities (e.g. &amp;, &quot;)
    cleaned = html.unescape(cleaned)
    # Apply standard whitespace normalization
    return normalize_text(cleaned)


def compute_content_fingerprint(text: str) -> str:
    """Compute a deterministic SHA-256 fingerprint over normalized text representation."""
    import hashlib

    normalized = " ".join(preprocess_record_text(text).lower().split())
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


def load_taxonomy_parents(
    path: str | Path = "taxonomy/taxonomy.json",
) -> dict[str, str | None]:
    """Load mapping of canonical label ID to parent canonical label ID (or None)."""
    tax_path = Path(path)
    with tax_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    labels = data.get("labels", [])
    return {
        label["id"]: label.get("parent")
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

    if not isinstance(record["id"], str) or not record["id"].strip() or "::" in record["id"]:
        raise ValueError("Invalid or empty ID.")

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

    # If direct labels are present, ensure canonical subset of closed labels
    labels_direct = record.get("labels_direct")
    if labels_direct is not None:
        if not isinstance(labels_direct, list) or len(labels_direct) == 0:
            raise ValueError(
                f"Record '{record['id']}' has invalid 'labels_direct': must be non-empty list."
            )
        for d_lbl in labels_direct:
            if allowed_labels is not None and d_lbl not in allowed_labels:
                raise ValueError(
                    f"Record '{record['id']}' direct label '{d_lbl}' not in canonical taxonomy."
                )
        if not set(labels_direct).issubset(set(labels)):
            raise ValueError(
                f"Record '{record['id']}' hierarchy labels must be a superset of labels_direct."
            )

    if record["language"] not in VALID_LANGUAGES:
        raise ValueError(
            f"Record '{record['id']}' has invalid language '{record['language']}'. "
            f"Allowed: {sorted(VALID_LANGUAGES)}"
        )

    VALID_SOURCES = {"cordis_h2020", "sib200", "synthetic"}
    if record["source"] not in VALID_SOURCES:
        raise ValueError(
            f"Record '{record['id']}' source '{record['source']}' is not supported. "
            f"Allowed: {sorted(VALID_SOURCES)}"
        )

    if not isinstance(record["license"], str) or not record["license"].strip():
        raise ValueError(f"Record '{record['id']}' has invalid 'license'.")

    if record["split"] not in VALID_SPLITS:
        raise ValueError(
            f"Record '{record['id']}' has invalid split '{record['split']}'. "
            f"Allowed: {sorted(VALID_SPLITS)}"
        )


def hierarchy_closure(labels: Iterable[str], parent_by_label: Mapping[str, str | None]) -> set[str]:
    """Compute the transitive ancestral closure of a set of labels."""
    closed: set[str] = set()
    for label in labels:
        current: str | None = label
        seen: set[str] = set()
        while current is not None and current not in seen:
            seen.add(current)
            closed.add(current)
            current = parent_by_label.get(current)
    return closed


def check_hierarchy_violations(
    labels: Iterable[str], parent_by_label: Mapping[str, str | None]
) -> list[tuple[str, str]]:
    """Return list of (child, missing_parent) pairs where child is predicted
    but parent is absent.
    """
    label_set = set(labels)
    violations: list[tuple[str, str]] = []
    for label in label_set:
        parent = parent_by_label.get(label)
        if parent is not None and parent not in label_set:
            violations.append((label, parent))
    return sorted(violations)


def hierarchy_violation_rate(
    predictions: Iterable[Iterable[str]],
    parent_by_label: Mapping[str, str | None],
) -> float:
    """Calculate the fraction of prediction instances that violate hierarchy consistency."""
    preds = list(predictions)
    if not preds:
        return 0.0
    violating = sum(1 for p in preds if len(check_hierarchy_violations(p, parent_by_label)) > 0)
    return violating / len(preds)


def process_cordis_h2020(
    raw_dir: str | Path = "data/raw/cordis_h2020",
    output_dir: str | Path = "data/processed/cordis_h2020/en",
    splits_dir: str | Path = "data/splits/cordis_h2020",
    taxonomy_path: str | Path = "taxonomy/taxonomy.json",
    label_map_path: str | Path = "taxonomy/label_map.json",
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    max_records: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Preprocess official CORDIS Horizon 2020 projects into canonical English splits.

    Extracts project objectives and EuroSciVoc classifications, applies active-label
    support filtering, closes labels along the EuroSciVoc hierarchy, detects duplicate
    objective texts to prevent train/validation/test contamination, and assigns
    leakage-free stratified project-level splits.
    """
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    split_path = Path(splits_dir)
    tax_path = Path(taxonomy_path)
    map_path = Path(label_map_path)

    project_csv = raw_path / "project.csv"
    euroscivoc_csv = raw_path / "euroSciVoc.csv"

    if not project_csv.exists():
        raise FileNotFoundError(f"Missing raw project file: {project_csv}")
    if not euroscivoc_csv.exists():
        raise FileNotFoundError(f"Missing raw EuroSciVoc file: {euroscivoc_csv}")

    print("Loading canonical taxonomy and label mapping...")
    allowed_labels = load_canonical_label_ids(tax_path)
    parent_by_label = load_taxonomy_parents(tax_path)
    label_map = load_label_map(map_path)

    with tax_path.open(encoding="utf-8") as f:
        tax_doc = json.load(f)
    tax_version = tax_doc.get("version", "0.4.0")
    direct_supported_codes = {
        lbl["euroscivoc_code"]
        for lbl in tax_doc.get("labels", [])
        if lbl.get("is_direct_supported", False)
    }

    # 1. Parse official EuroSciVoc classifications
    print("Reading EuroSciVoc classifications...")
    import csv

    proj_direct_codes: dict[str, set[str]] = defaultdict(set)
    proj_direct_titles: dict[str, set[str]] = defaultdict(set)
    with open(euroscivoc_csv, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader, None)
        for row in reader:
            if not row or len(row) < 5:
                continue
            code, _path, title, _desc, pid = (
                row[0].strip(),
                row[1].strip(),
                row[2].strip(),
                row[3].strip(),
                row[4].strip(),
            )
            if pid and code:
                proj_direct_codes[pid].add(code)
                proj_direct_titles[pid].add(title)

    # 2. Parse Projects with robust unescaped semicolon reconstruction
    print("Reading and cleaning CORDIS H2020 projects...")
    raw_projects: list[dict[str, str]] = []
    normal_rows = 0
    repaired_rows = 0
    malformed_rows = 0

    with open(project_csv, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader, None)  # Skip header
        expected_cols = 22

        for row in reader:
            if not row:
                continue
            if len(row) == expected_cols:
                normal_rows += 1
                row_fixed = row
            elif len(row) > expected_cols:
                # Field 15 is objective; unescaped semicolons inside text
                obj = ";".join(row[15 : len(row) - 6])
                row_fixed = row[:15] + [obj] + row[len(row) - 6 :]
                if len(row_fixed) == expected_cols:
                    repaired_rows += 1
                else:
                    malformed_rows += 1
                    continue
            else:
                malformed_rows += 1
                continue

            raw_projects.append(
                {
                    "id": row_fixed[0].strip(),
                    "title": row_fixed[3].strip(),
                    "objective": row_fixed[15].strip(),
                    "human_validated": row_fixed[20].strip() or "NA",
                }
            )

    print(
        f"Parsed {len(raw_projects)} projects from project.csv "
        f"({normal_rows} normal, {repaired_rows} repaired, {malformed_rows} malformed dropped)."
    )

    clean_records: list[dict[str, Any]] = []
    skipped_no_objective = 0
    skipped_no_active_labels = 0

    for proj in raw_projects:
        pid = proj["id"]
        raw_obj = proj["objective"]
        title = proj["title"]
        human_val = proj["human_validated"]

        if not raw_obj:
            skipped_no_objective += 1
            continue

        clean_text = preprocess_record_text(raw_obj)
        if not clean_text:
            skipped_no_objective += 1
            continue

        direct_codes = proj_direct_codes.get(pid, set())
        # Filter direct codes to only those meeting direct support threshold
        supported_codes_for_proj = {c for c in direct_codes if c in direct_supported_codes}
        # Map direct supported codes to active canonical taxonomy labels
        direct_canonical = sorted(
            list({label_map[c] for c in supported_codes_for_proj if c in label_map})
        )

        if not direct_canonical:
            # All direct labels were below support threshold or project unclassified
            skipped_no_active_labels += 1
            continue

        # Compute hierarchy closure: direct + all active ancestors
        closed_canonical = sorted(list(hierarchy_closure(direct_canonical, parent_by_label)))
        fp = compute_content_fingerprint(clean_text)

        rec = {
            "id": f"cordis:{pid}:en",
            "project_id": pid,
            "pair_id": f"cordis:{pid}",
            "source_id": pid,
            "title": title,
            "text": clean_text,
            "labels_direct": direct_canonical,
            "labels": closed_canonical,
            "language": "en",
            "source": "cordis_h2020",
            "license": "CC BY 4.0",
            "taxonomy_version": tax_version,
            "content_fingerprint": fp,
            "human_validated": human_val,
            "synthetic_language_variant": False,
            "source_label_ids": sorted(list(supported_codes_for_proj)),
            "source_labels": sorted(list(proj_direct_titles.get(pid, set()))),
        }
        clean_records.append(rec)

    print(
        f"Extracted {len(clean_records)} usable labeled English project records "
        f"(skipped: {skipped_no_objective} empty objective, "
        f"{skipped_no_active_labels} unsupported/unlabeled)."
    )

    if max_records and max_records < len(clean_records):
        clean_records = clean_records[:max_records]
        print(f"Subsampled to {len(clean_records)} records as requested by max_records.")

    # 3. Stratified Partitioning Grouped by Content Fingerprint
    # Grouping by content_fingerprint strictly prevents exact duplicate text leakage across splits.
    print("Performing multi-label stratified split grouped by content fingerprint...")
    train_recs, val_recs, test_recs = grouped_multilabel_stratified_split(
        records=clean_records,
        group_key="content_fingerprint",
        train_ratio=split_ratios[0],
        validation_ratio=split_ratios[1],
        test_ratio=split_ratios[2],
        seed=seed,
    )

    splits_dict: dict[str, list[dict[str, Any]]] = {
        "train": train_recs,
        "validation": val_recs,
        "test": test_recs,
    }

    # Assign split attribute and validate each record
    project_split_map: dict[str, str] = {}
    out_path.mkdir(parents=True, exist_ok=True)
    split_path.mkdir(parents=True, exist_ok=True)

    for split_name, recs in splits_dict.items():
        for r in recs:
            r["split"] = split_name
            validate_record(r, allowed_labels=allowed_labels)
            project_split_map[r["project_id"]] = split_name

        split_file = out_path / f"{split_name}.jsonl"
        save_jsonl(recs, split_file)
        print(f"Saved {len(recs)} records -> {split_file}")

    # Save project split manifest
    split_map_file = split_path / "project_splits.json"
    with open(split_map_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": seed,
                "split_ratios": list(split_ratios),
                "total_projects": len(project_split_map),
                "project_to_split": project_split_map,
            },
            f,
            indent=2,
        )
    print(f"Saved project split map -> {split_map_file}")

    # Create reproducibility manifest for English benchmark
    create_manifest(
        benchmark_name="cordis_h2020_en",
        output_dir=out_path,
        generation_parameters={
            "seed": seed,
            "split_ratios": list(split_ratios),
            "max_records": max_records,
            "min_direct_support": tax_doc.get("min_direct_support_threshold", 50),
        },
        source_metadata={
            "source": "official CORDIS / European Union open data release",
            "source_title": "CORDIS - EU research projects under Horizon 2020 (2014-2020)",
            "raw_dir": str(raw_path),
            "raw_project_csv": "project.csv",
            "raw_euroscivoc_csv": "euroSciVoc.csv",
        },
        taxonomy_dir=tax_path.parent,
    )

    return splits_dict
