"""Release integrity and project-level leakage checks shared by training and the CLI."""

from __future__ import annotations

import hashlib
import json
import re
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
CODE_SWITCH_STRATEGIES = ("sentence_mix", "chunk_mix")
TRANSLATION_PROVENANCE = ("translation_backend", "translation_model", "translation_model_revision")

Partitions = dict[str, list[dict[str, Any]]]


def check_disjoint_partitions(partitions: dict[str, list[dict[str, Any]]]) -> None:
    """Keep record IDs, project groups, and normalized content in one partition.

    Missing or empty values (an unpaired row's pair_id) are not identities and are skipped.
    """
    for first, second in combinations(partitions, 2):
        for key in ("id", "pair_id", "project_id", "text"):

            def values(split: str, key: str = key) -> set[str]:
                return {
                    compute_content_fingerprint(r["text"]) if key == "text" else r[key]
                    for r in partitions[split]
                    if r.get(key)
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


def _check_languages(languages: Sequence[str]) -> None:
    if not languages or set(languages) - {"en", "tr"} or len(set(languages)) != len(languages):
        raise ValueError("Select unique CORDIS training languages from en and tr.")


def _direct_labels(taxonomy_path: Path) -> set[str]:
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    return {label["id"] for label in taxonomy["labels"] if label.get("is_direct_supported")}


def _load_splits(directory: Path, taxonomy_path: Path) -> Partitions:
    return {s: load_audit_records(directory / f"{s}.jsonl", taxonomy_path) for s in SPLITS}


def _audit_english(english: Partitions, split_map: dict[str, str] | None) -> None:
    """Check canonical identity, fingerprints and project grouping; the split map when known."""
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
            if pid in seen_projects or (split_map is not None and split_map.get(pid) != split):
                raise ValueError("CORDIS duplicate project or split-map mismatch.")
            seen_projects.add(pid)
            if r.get("content_fingerprint") != compute_content_fingerprint(r["text"]):
                raise ValueError("CORDIS English content fingerprint mismatch.")
    if split_map is not None and seen_projects != set(split_map):
        raise ValueError("CORDIS English corpus and project split map differ.")
    check_disjoint_partitions(english)


def _check_full_support(english: Partitions, direct_labels: set[str]) -> None:
    modeled = {label for r in english["train"] for label in r["labels_direct"]}
    if modeled != direct_labels:
        raise ValueError(
            f"CORDIS training is missing supported labels: {sorted(direct_labels - modeled)}"
        )


def _check_translated_record(
    r: dict[str, Any], source: dict[str, Any], *, strict: bool
) -> set[str]:
    """Raise on broken invariants; with strict=False, return tolerated alignment drift tags."""
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
    alignment = r.get("sentence_alignment", [])
    if (
        not isinstance(alignment, list)
        or not alignment
        or any(
            not isinstance(p, dict)
            or type(p.get("index")) is not int
            or any(not isinstance(p.get(k), str) or not p[k].strip() for k in ("en", "tr"))
            for p in alignment
        )
    ):
        raise ValueError("CORDIS sentence alignment differs from source/output.")
    english_parts = [p["en"] for p in alignment]
    drift = set()
    if english_parts != segment_sentences(source["text"]):
        drift.add("segmenter_drift")
    if " ".join(p["tr"] for p in alignment) != r["text"]:
        drift.add("alignment_text_mismatch")
    # Imported releases can use older sentence boundaries. Preserve them when they cover
    # the exact source in order, rather than requiring the current segmenter's boundaries.
    # Parts that match the segmenter are accepted as is: it rewrites bullet and list marks.
    legacy_boundaries = " ".join(english_parts) == source["text"]
    if (
        [p["index"] for p in alignment] != list(range(len(alignment)))
        or (
            "segmenter_drift" in drift
            and _without_space("".join(english_parts)) != _without_space(source["text"])
        )
        or (strict and "alignment_text_mismatch" in drift)
        or (strict and "segmenter_drift" in drift and not legacy_boundaries)
    ):
        raise ValueError("CORDIS sentence alignment differs from source/output.")
    return drift


def _without_space(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _audit_translation(
    english: Partitions,
    translated: Partitions,
    *,
    lock: dict[str, Any] | None,
    allow_mock: bool,
    strict: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check EN/TR invariants; return the shared provenance and any tolerated issues."""
    observed = set()
    drift_count = 0
    mismatched, qa_failed = [], []
    for split, records in translated.items():
        sources = {r["project_id"]: r for r in english[split]}
        if len(records) != len(sources) or {r["project_id"] for r in records} != set(sources):
            raise ValueError(f"Incomplete CORDIS EN/TR alignment in {split}.")
        for r in records:
            source = sources[r["project_id"]]
            drift = _check_translated_record(r, source, strict=strict)
            drift_count += "segmenter_drift" in drift
            if "alignment_text_mismatch" in drift:
                mismatched.append(r["id"])
            provenance = tuple(r.get(key) for key in TRANSLATION_PROVENANCE)
            observed.add(provenance)
            if not allow_mock and (
                provenance[0] != "huggingface_local"
                or (lock is not None and provenance[1:] != (lock["model"], lock["revision"]))
            ):
                raise ValueError(
                    "CORDIS production translation provenance mismatch or mock output."
                )
            ok, issues = check_translation_qa(source["text"], r["text"])
            if not ok and not allow_mock:
                if strict:
                    raise ValueError(f"CORDIS translation QA failed: {issues}")
                qa_failed.append(r["id"])
    if len(observed) != 1:
        raise ValueError("CORDIS translation provenance is inconsistent across records.")
    check_disjoint_partitions(translated)
    report = {
        "segmenter_drift": drift_count,
        "alignment_text_mismatch": mismatched,
        "qa_failed": qa_failed,
    }
    return dict(zip(TRANSLATION_PROVENANCE, observed.pop(), strict=True)), report


def _check_cross_language(corpora: dict[str, Partitions], languages: Sequence[str]) -> None:
    check_disjoint_partitions(
        {s: [r for language in languages for r in corpora[language][s]] for s in SPLITS}
    )


def _split_sha256(base: Path, languages: Sequence[str]) -> dict[str, dict[str, str]]:
    return {
        language: {s: compute_file_sha256(base / language / f"{s}.jsonl") for s in SPLITS}
        for language in languages
    }


def _check_legacy_outputs(
    language: str, manifest: Any, records: Partitions, hashes: dict[str, str]
) -> None:
    """Match each split to the sha256 and any record count a legacy manifest declares for it."""
    outputs = manifest.get("output_files") if isinstance(manifest, dict) else None
    for split in SPLITS:
        meta = outputs.get(f"{split}.jsonl") if isinstance(outputs, dict) else None
        if not isinstance(meta, dict) or "sha256" not in meta:
            raise ValueError(f"CORDIS {language} manifest lists no sha256 for {split}.jsonl.")
        if meta["sha256"] != hashes[split] or (
            "record_count" in meta and meta["record_count"] != len(records[split])
        ):
            raise ValueError(f"CORDIS {language}/{split}.jsonl differs from its manifest.")


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
    _check_languages(languages)
    direct_labels = _direct_labels(taxonomy_path)
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
    english = _load_splits(en_directory, taxonomy_path)
    _audit_english(english, split_map)
    if require_full_support:
        _check_full_support(english, direct_labels)
    corpora, manifests = {"en": english}, {"en": en_manifest}
    if "tr" in languages:
        directory = base / "tr"
        manifest = validate_manifest(
            directory, "cordis_h2020_tr", [f"{s}.jsonl" for s in SPLITS], taxonomy_path.parent
        )
        check_input_inventory(directory, manifest, [en_directory / f"{s}.jsonl" for s in SPLITS])
        translated = _load_splits(directory, taxonomy_path)
        _audit_translation(
            english, translated, lock=get_source_config("translation"), allow_mock=allow_mock
        )
        corpora["tr"], manifests["tr"] = translated, manifest
    _check_cross_language(corpora, languages)
    return {
        "corpora": corpora,
        "manifests": manifests,
        "label_universe": sorted(direct_labels),
        "split_sha256": _split_sha256(base, list(corpora)),
    }


def validate_prebuilt_cordis_release(
    base: str | Path = REPO_ROOT / "data/processed/cordis_h2020",
    languages: Sequence[str] = ("en", "tr"),
    *,
    taxonomy_path: str | Path = REPO_ROOT / "taxonomy/taxonomy.json",
    require_full_support: bool = True,
    allow_mock: bool = False,
) -> dict[str, Any]:
    """Audit a distributed release whose manifests predate the raw-source locks.

    Every record-level invariant of validate_cordis_release still holds, every audited language
    needs a manifest, and each split must match the sha256 (and any record count) it declares.
    The manifest schema, raw CSV checksums, split map and translation-model lock cannot be
    verified, so the observed translation provenance and split hashes are returned for the
    training record instead. Alignment/text mismatches and QA failures are returned as
    translation_issues: reported, not enforced.
    """
    base, taxonomy_path = Path(base), Path(taxonomy_path)
    _check_languages(languages)
    direct_labels = _direct_labels(taxonomy_path)
    english = _load_splits(base / "en", taxonomy_path)
    _audit_english(english, split_map=None)
    if require_full_support:
        _check_full_support(english, direct_labels)
    corpora = {"en": english}
    translation, issues = None, None
    if "tr" in languages:
        corpora["tr"] = _load_splits(base / "tr", taxonomy_path)
        translation, issues = _audit_translation(
            english, corpora["tr"], lock=None, allow_mock=allow_mock, strict=False
        )
    _check_cross_language(corpora, languages)
    hashes = _split_sha256(base, list(corpora))
    manifests = {}
    for language in corpora:
        path = base / language / "manifest.json"
        if not path.is_file():
            raise ValueError(f"CORDIS {language} release has no manifest.json to lock its splits.")
        manifests[language] = json.loads(path.read_text(encoding="utf-8"))
        _check_legacy_outputs(language, manifests[language], corpora[language], hashes[language])
    return {
        "corpora": corpora,
        "manifests": manifests,
        "label_universe": sorted(direct_labels),
        "split_sha256": hashes,
        "audit": "prebuilt_records_only",
        "translation_provenance": translation,
        "translation_issues": issues,
    }


def validate_code_switch_release(
    base: str | Path,
    release: dict[str, Any],
    strategies: Sequence[str] = CODE_SWITCH_STRATEGIES,
    *,
    taxonomy_path: str | Path = REPO_ROOT / "taxonomy/taxonomy.json",
) -> dict[str, Any]:
    """Verify mixed evaluation records against their exact audited EN/TR source pairs."""
    from naltra.data.code_switching import create_code_switched_record, pair_aligned_records

    base, taxonomy_path = Path(base), Path(taxonomy_path)
    if (
        not strategies
        or len(set(strategies)) != len(strategies)
        or set(strategies) - set(CODE_SWITCH_STRATEGIES)
    ):
        raise ValueError("Select unique sentence_mix or chunk_mix code-switch tracks.")
    if not {"en", "tr"} <= set(release["corpora"]):
        raise ValueError("Code-switch auditing requires both audited English and Turkish sources.")
    manifests, corpora = {}, {}
    for strategy in strategies:
        directory = base / "code_switch" / strategy / "balanced"
        manifest = validate_manifest(
            directory,
            "cordis_h2020_code_switch",
            ["validation.jsonl", "test.jsonl"],
            taxonomy_path.parent,
        )
        check_input_inventory(
            directory,
            manifest,
            [
                base / language / f"{split}.jsonl"
                for language in ("en", "tr")
                for split in ("validation", "test")
            ],
        )
        if manifest["generation_parameters"] != {
            "strategy": strategy,
            "strength": "balanced",
            "base_seed": 42,
            "splits": ["validation", "test"],
        }:
            raise ValueError("Code-switch parameters differ from the benchmark specification.")
        partitions = {}
        for split in ("validation", "test"):
            pairs = pair_aligned_records(
                release["corpora"]["en"][split], release["corpora"]["tr"][split]
            )
            expected = [create_code_switched_record(en, tr, strategy=strategy) for en, tr in pairs]
            actual = load_audit_records(directory / f"{split}.jsonl", taxonomy_path)
            if actual != expected:
                raise ValueError(f"Code-switch {strategy}/{split} changed source fidelity.")
            partitions[split] = actual
        check_disjoint_partitions({"train": release["corpora"]["en"]["train"], **partitions})
        corpora[strategy], manifests[strategy] = partitions, manifest
    return {"corpora": corpora, "manifests": manifests}
