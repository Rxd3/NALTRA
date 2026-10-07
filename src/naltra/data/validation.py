"""Release integrity and project-level leakage checks shared by training and the CLI."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

from naltra.data.loader import load_jsonl
from naltra.data.manifest import (
    REPO_ROOT,
    compute_file_sha256,
    get_source_config,
    validate_manifest,
)
from naltra.data.preprocessing import (
    compute_content_fingerprint,
    hierarchy_closure,
    validate_record,
)
from naltra.data.translation import check_translation_qa, segment_sentences

SPLITS = ("train", "validation", "test")


def check_disjoint_partitions(partitions: dict[str, list[dict[str, Any]]]) -> None:
    """Keep record IDs, project groups, and normalized content in one partition."""
    for first, second in combinations(partitions, 2):
        for key in ("id", "pair_id", "project_id", "text"):

            def values(split: str, key: str = key) -> set[str]:
                return {
                    compute_content_fingerprint(r["text"]) if key == "text" else r[key]
                    for r in partitions[split]
                    if key in r
                }

            if values(first) & values(second):
                raise ValueError(f"{key} contamination between {first} and {second}.")


def load_audit_records(path: Path, taxonomy_path: Path | None = None) -> list[dict[str, Any]]:
    taxonomy_path = taxonomy_path or REPO_ROOT / "taxonomy/taxonomy.json"
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    parents = {label["id"]: label["parent"] for label in taxonomy["labels"]}
    direct = {label["id"] for label in taxonomy["labels"] if label.get("is_direct_supported")}
    records = load_jsonl(path)
    if not records:
        raise ValueError(f"Empty release partition: {path}.")
    seen = set()
    for record in records:
        validate_record(record, allowed_labels=set(parents))
        if record["split"] != path.stem:
            raise ValueError(f"{path}: physical split mismatch for {record['id']}.")
        if record["id"] in seen:
            raise ValueError(f"{path}: duplicate record ID {record['id']}.")
        seen.add(record["id"])
        if record["source"] == "cordis_h2020":
            if record.get("taxonomy_version") != taxonomy["version"]:
                raise ValueError("CORDIS record taxonomy version mismatch.")
            if not set(record["labels_direct"]) <= direct:
                raise ValueError("CORDIS direct label is outside the supported label universe.")
            if set(record["labels"]) != hierarchy_closure(record["labels_direct"], parents):
                raise ValueError("CORDIS labels must equal the exact ancestor closure.")
    return records


def check_input_inventory(directory: Path, manifest: dict, expected: Sequence[Path]) -> None:
    actual = {(directory / path).resolve() for path in manifest["input_files"]}
    if actual != {path.resolve() for path in expected}:
        raise ValueError(f"{directory}: incomplete or unexpected source inventory.")


def validate_cordis_release(
    base: str | Path = REPO_ROOT / "data/processed/cordis_h2020",
    languages: Sequence[str] = ("en", "tr"),
    *,
    taxonomy_path: str | Path = REPO_ROOT / "taxonomy/taxonomy.json",
    require_full_support: bool = True,
    allow_mock: bool = False,
) -> dict[str, Any]:
    """Audit clean corpora before training; always verify their canonical EN source."""
    base, taxonomy_path = Path(base), Path(taxonomy_path)
    if not languages or set(languages) - {"en", "tr"} or len(set(languages)) != len(languages):
        raise ValueError("Select unique CORDIS training languages from en and tr.")
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    direct_labels = {
        label["id"] for label in taxonomy["labels"] if label.get("is_direct_supported")
    }
    manifests, corpora = {}, {}
    lock = get_source_config("cordis_h2020")
    en_directory = base / "en"
    en_manifest = validate_manifest(
        en_directory, "cordis_h2020_en", [f"{s}.jsonl" for s in SPLITS], taxonomy_path.parent
    )
    for key, value in lock.items():
        if en_manifest["source_metadata"].get(key) != value:
            raise ValueError(f"CORDIS source lock differs for {key}.")
    raw = Path(en_manifest["source_metadata"]["raw_dir"])
    if not raw.is_absolute():
        raw = en_directory / raw
    split_map_path = Path(en_manifest["source_metadata"]["split_map"])
    if not split_map_path.is_absolute():
        split_map_path = en_directory / split_map_path
    check_input_inventory(
        en_directory,
        en_manifest,
        [
            raw / "project.csv",
            raw / "euroSciVoc.csv",
            taxonomy_path.parent / "label_map.json",
            split_map_path,
        ],
    )
    for filename, key in (
        ("project.csv", "project_sha256"),
        ("euroSciVoc.csv", "euroscivoc_sha256"),
    ):
        if compute_file_sha256(raw / filename) != lock[key]:
            raise ValueError(f"CORDIS raw source mismatch: {filename}.")
    split_map = json.loads(split_map_path.read_text(encoding="utf-8"))["project_to_split"]
    english = {
        split: load_audit_records(en_directory / f"{split}.jsonl", taxonomy_path)
        for split in SPLITS
    }
    seen_projects = set()
    for split, records in english.items():
        for r in records:
            pid = r["project_id"]
            if (
                r["language"] != "en"
                or r["id"] != f"cordis:{pid}:en"
                or r["source"] != "cordis_h2020"
            ):
                raise ValueError("Invalid canonical CORDIS English identity.")
            if pid in seen_projects or split_map.get(pid) != split:
                raise ValueError("CORDIS duplicate project or split-map mismatch.")
            seen_projects.add(pid)
            if r.get("content_fingerprint") != compute_content_fingerprint(r["text"]):
                raise ValueError("CORDIS English content fingerprint mismatch.")
    if seen_projects != set(split_map):
        raise ValueError("CORDIS English corpus and project split map differ.")
    check_disjoint_partitions(english)
    if require_full_support:
        modeled = {label for r in english["train"] for label in r["labels_direct"]}
        if modeled != direct_labels:
            raise ValueError(
                f"CORDIS training is missing supported labels: {sorted(direct_labels - modeled)}"
            )
    corpora["en"], manifests["en"] = english, en_manifest
    if "tr" in languages:
        directory = base / "tr"
        manifest = validate_manifest(
            directory, "cordis_h2020_tr", [f"{s}.jsonl" for s in SPLITS], taxonomy_path.parent
        )
        check_input_inventory(directory, manifest, [en_directory / f"{s}.jsonl" for s in SPLITS])
        translated = {
            s: load_audit_records(directory / f"{s}.jsonl", taxonomy_path) for s in SPLITS
        }
        translation_lock = get_source_config("translation")
        for split, records in translated.items():
            sources = {r["project_id"]: r for r in english[split]}
            if len(records) != len(sources) or {r["project_id"] for r in records} != set(sources):
                raise ValueError(f"Incomplete CORDIS EN/TR alignment in {split}.")
            for r in records:
                source = sources[r["project_id"]]
                for key in ("labels", "labels_direct", "pair_id", "split", "source", "license"):
                    if r[key] != source[key]:
                        raise ValueError(f"CORDIS translation changed invariant {key}.")
                if (
                    r["language"] != "tr"
                    or r["id"] != f"cordis:{r['project_id']}:tr"
                    or r.get("variant_of") != source["id"]
                ):
                    raise ValueError("CORDIS translation identity mismatch.")
                source_hash = hashlib.sha256(source["text"].encode("utf-8")).hexdigest()
                if r.get("translation_source_hash") != source_hash:
                    raise ValueError("CORDIS translation source hash mismatch.")
                if not allow_mock and (
                    r.get("translation_backend") != "huggingface_local"
                    or r.get("translation_model") != translation_lock["model"]
                    or r.get("translation_model_revision") != translation_lock["revision"]
                ):
                    raise ValueError(
                        "CORDIS production translation provenance mismatch or mock output."
                    )
                alignment = r.get("sentence_alignment", [])
                if (
                    not alignment
                    or [p.get("index") for p in alignment] != list(range(len(alignment)))
                    or [p.get("en") for p in alignment] != segment_sentences(source["text"])
                    or " ".join(p.get("tr", "") for p in alignment) != r["text"]
                ):
                    raise ValueError("CORDIS sentence alignment differs from source/output.")
                ok, issues = check_translation_qa(source["text"], r["text"])
                if not ok and not allow_mock:
                    raise ValueError(f"CORDIS translation QA failed: {issues}")
        check_disjoint_partitions(translated)
        corpora["tr"], manifests["tr"] = translated, manifest
    check_disjoint_partitions(
        {s: [r for language in languages for r in corpora[language][s]] for s in SPLITS}
    )
    return {"corpora": corpora, "manifests": manifests, "label_universe": sorted(direct_labels)}
