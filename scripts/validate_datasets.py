"""Comprehensive dataset-wide verification and auditing script for NALTRA."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Ensure src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

from naltra.data.code_switching import (  # noqa: E402
    create_code_switched_record,
    pair_aligned_records,
)
from naltra.data.loader import load_jsonl  # noqa: E402
from naltra.data.manifest import get_source_config, validate_manifest  # noqa: E402
from naltra.data.noise import create_noisy_record  # noqa: E402
from naltra.data.ood import (  # noqa: E402
    MASSIVE_ALLOWED_SCENARIOS,
    validate_ood_record,
)
from naltra.data.preprocessing import (  # noqa: E402
    compute_content_fingerprint,
    load_canonical_label_ids,
    validate_record,
)


def load_audit_records(path: Path) -> list[dict[str, Any]]:
    """Check schema, unique IDs and the physical partition before indexing records."""
    records = load_jsonl(path)
    expected_split = path.stem.split("_")[0]
    seen = set()
    labels = load_canonical_label_ids(REPO_ROOT / "taxonomy/taxonomy.json")
    for record in records:
        if path.stem.endswith("_ood"):
            validate_ood_record(record)
        else:
            validate_record(record, allowed_labels=labels)
        if record["split"] != expected_split:
            raise ValueError(f"{path}: {record['id']} split mismatch; expected {expected_split}.")
        if record["id"] in seen:
            raise ValueError(f"{path}: duplicate record ID {record['id']}.")
        seen.add(record["id"])
    return records


def check_manifest(directory: Path, name: str, files: list[str]) -> dict[str, Any]:
    manifest = validate_manifest(directory, name, files, REPO_ROOT / "taxonomy")
    expected_parameters: dict[str, Any] = {}
    if name.startswith("noisy_"):
        expected_parameters = {
            "strategy": "combined",
            "severity": "medium",
            "base_seed": 42,
            "splits": ["validation", "test"],
        }
    elif name == "code_switch_sib200":
        expected_parameters = {
            "strategy": "chunk_mix",
            "strength": "balanced",
            "base_seed": 42,
            "splits": ["validation", "test"],
        }
    elif name == "clean_mn_ds":
        expected_parameters = {
            "seed": 42,
            "train_ratio": 0.7,
            "validation_ratio": 0.15,
            "test_ratio": 0.15,
            "split_method": "content_grouped_multilabel_stratification",
        }
    elif name.startswith("near_ood_sib200_"):
        expected_parameters = {"heldout_topic": name.removeprefix("near_ood_sib200_")}
    elif name == "far_ood_massive":
        expected_parameters = {
            "seed": 42,
            "val_sample_size": 250,
            "test_sample_size": 500,
            "allowed_scenarios": sorted(MASSIVE_ALLOWED_SCENARIOS),
            "overlap_policy": "preserve_test_filter_validation_content",
        }
    for key, value in expected_parameters.items():
        if manifest["generation_parameters"].get(key) != value:
            raise ValueError(
                f"{name}: generation parameter {key} differs from this benchmark specification."
            )
    if name.startswith("clean_") or name == "far_ood_massive":
        source_name = {
            "clean_sib200": "sib200",
            "clean_multifin_official": "multifin",
            "clean_mn_ds": "mn_ds",
            "far_ood_massive": "massive",
        }[name]
        for key, value in get_source_config(source_name).items():
            if manifest["source_metadata"].get(key) != value:
                raise ValueError(f"{name}: source {key} differs from the versioned source lock.")
    # Require the complete source inventory; a manifest cannot omit a changed input.
    if name in ("clean_sib200", "clean_multifin_official"):
        inputs = [REPO_ROOT / "taxonomy/label_map.json", REPO_ROOT / "configs/data.yaml"]
    elif name == "clean_mn_ds":
        inputs = [
            REPO_ROOT / "data/raw/mn_ds/MN-DS-news-classification.csv",
            REPO_ROOT / "taxonomy/label_map.json",
        ]
    elif name == "far_ood_massive":
        inputs = [
            REPO_ROOT / "data/raw/massive" / f"{locale}.jsonl" for locale in ("en-US", "tr-TR")
        ]
    else:
        if name == "noisy_robustness_benchmark":
            datasets = ("sib200", "multifin", "mn_ds")
        elif name == "multifin_leakage_free":
            datasets = ("multifin",)
        elif name.startswith("noisy_"):
            datasets = (name.removeprefix("noisy_"),)
        else:
            datasets = ("sib200",)
        splits = (
            ("train", "validation", "test")
            if name.startswith("near_ood_") or name == "multifin_leakage_free"
            else ("validation", "test")
        )
        inputs = [
            REPO_ROOT / "data/processed" / dataset / f"{split}.jsonl"
            for dataset in datasets
            for split in splits
        ]
    actual_inputs = {(directory / p).resolve() for p in manifest["input_files"]}
    if actual_inputs != {p.resolve() for p in inputs}:
        raise ValueError(f"{name}: manifest source inventory is incomplete or unexpected.")
    print(f"  [OK] {name} manifest hashes, counts, taxonomy, sources and code verified")
    return manifest


def check_disjoint_partitions(partitions: dict[str, list[dict[str, Any]]]) -> None:
    """Reject reused IDs, aligned pairs or normalized content across partitions."""
    seen: dict[tuple[str, str], str] = {}
    for split, records in partitions.items():
        for record in records:
            keys = [("id", record["id"]), ("content", compute_content_fingerprint(record["text"]))]
            if record.get("pair_id"):
                keys.append(("pair", record["pair_id"]))
            for key in keys:
                if key in seen and seen[key] != split:
                    raise ValueError(
                        f"{key[0]} contamination between {seen[key]} and {split}: {key[1]}"
                    )
                seen[key] = split


def print_section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Track 1: Clean SIB-200 Benchmark
# ---------------------------------------------------------------------------
def validate_sib200(canonical_labels: set[str]) -> bool:
    print_section("1. SIB-200 Cross-Language Topic Benchmark (Clean)")
    sib_dir = REPO_ROOT / "data" / "processed" / "sib200"
    check_manifest(sib_dir, "clean_sib200", [f"{s}.jsonl" for s in ("train", "validation", "test")])

    expected_counts = {"train": 1402, "validation": 198, "test": 408}
    expected_en = {"train": 701, "validation": 99, "test": 204}
    expected_tr = {"train": 701, "validation": 99, "test": 204}

    all_ids = set()
    pair_splits: dict[str, set[str]] = defaultdict(set)
    pair_labels: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    pair_langs: dict[str, set[str]] = defaultdict(set)
    pair_lang_counts: dict[str, Counter[str]] = defaultdict(Counter)
    missing_pair_ids = 0
    split_ids: dict[str, set[str]] = {}
    split_fingerprints: dict[str, set[str]] = {}

    passed = True

    for split_name, expected_total in expected_counts.items():
        file_path = sib_dir / f"{split_name}.jsonl"
        if not file_path.exists():
            print(f"[FAIL] Missing file: {file_path}")
            return False

        records = load_audit_records(file_path)
        split_ids[split_name] = {r["id"] for r in records}
        split_fingerprints[split_name] = {compute_content_fingerprint(r["text"]) for r in records}
        en_count = sum(1 for r in records if r["language"] == "en")
        tr_count = sum(1 for r in records if r["language"] == "tr")

        if (
            len(records) != expected_total
            or en_count != expected_en[split_name]
            or tr_count != expected_tr[split_name]
        ):
            print(
                f"[FAIL] Split '{split_name}' size mismatch: got {len(records)} "
                f"({en_count} en, {tr_count} tr), expected {expected_total}"
            )
            passed = False
        else:
            print(
                f"  [OK] Split '{split_name}': {len(records)} records "
                f"({en_count} en, {tr_count} tr)"
            )

        for r in records:
            all_ids.add(r["id"])

            try:
                validate_record(r, allowed_labels=canonical_labels)
            except Exception as exc:
                rec_id = r.get("id", "unknown")
                print(f"[FAIL] SIB-200 record {rec_id} schema validation failed: {exc}")
                passed = False

            if r.get("split") != split_name:
                print(
                    f"[FAIL] SIB-200 record {r.get('id')} split mismatch: "
                    f"got '{r.get('split')}', expected '{split_name}'"
                )
                passed = False

            p_id = r.get("pair_id")
            if not p_id or not isinstance(p_id, str) or not p_id.strip():
                print(f"[FAIL] SIB-200 record {r.get('id')} is missing required pair_id")
                missing_pair_ids += 1
                passed = False
            else:
                pair_splits[p_id].add(split_name)
                pair_labels[p_id].add(tuple(sorted(r["labels"])))
                pair_langs[p_id].add(r["language"])
                pair_lang_counts[p_id][r["language"]] += 1

            for lbl in r.get("labels", []):
                if lbl not in canonical_labels:
                    print(f"[FAIL] Record {r['id']} has unknown label: {lbl}")
                    passed = False

    expected_sum = sum(expected_counts.values())
    if len(all_ids) != expected_sum:
        print(f"[FAIL] Unique ID count mismatch: got {len(all_ids)}, expected {expected_sum}")
        passed = False
    else:
        print(f"  [OK] Unique IDs: {len(all_ids)} / {expected_sum} (zero duplicates)")

    # Split ID leakage
    t_v = split_ids["train"] & split_ids["validation"]
    t_te = split_ids["train"] & split_ids["test"]
    v_te = split_ids["validation"] & split_ids["test"]
    if t_v or t_te or v_te:
        print(
            f"[FAIL] Split ID leakage in SIB-200: "
            f"t_v={len(t_v)}, t_te={len(t_te)}, v_te={len(v_te)}"
        )
        passed = False
    else:
        print("  [OK] Split ID leakage: 0 (zero ID overlap between train, val, and test)")

    # Content-fingerprint leakage
    fp_tv = split_fingerprints["train"] & split_fingerprints["validation"]
    fp_tte = split_fingerprints["train"] & split_fingerprints["test"]
    fp_vte = split_fingerprints["validation"] & split_fingerprints["test"]
    if fp_tv or fp_tte or fp_vte:
        print(
            f"[FAIL] Content-fingerprint leakage in SIB-200: "
            f"t_v={len(fp_tv)}, t_te={len(fp_tte)}, v_te={len(fp_vte)}"
        )
        passed = False
    else:
        print("  [OK] Content-fingerprint leakage: 0 (zero content overlap across splits)")

    # Pair alignment
    misaligned_split = sum(1 for s in pair_splits.values() if len(s) != 1)
    misaligned_labels = sum(1 for labels in pair_labels.values() if len(labels) != 1)
    incomplete_pairs = sum(1 for langs in pair_langs.values() if langs != {"en", "tr"})
    non_one_to_one_pairs = sum(
        1
        for counts in pair_lang_counts.values()
        if counts["en"] != 1 or counts["tr"] != 1 or len(counts) != 2
    )

    if missing_pair_ids > 0:
        passed = False

    if misaligned_split or misaligned_labels or incomplete_pairs or non_one_to_one_pairs:
        print(
            f"[FAIL] Pair alignment issues: split_diff={misaligned_split}, "
            f"label_diff={misaligned_labels}, incomplete={incomplete_pairs}, "
            f"non_1to1={non_one_to_one_pairs}"
        )
        passed = False
    else:
        num_pairs = len(pair_lang_counts)
        print(f"  [OK] English/Turkish pair alignment: {num_pairs} pairs aligned (1 EN, 1 TR)")

    return passed


# ---------------------------------------------------------------------------
# Track 2: MultiFin Multilingual Multi-Label Benchmark (Official Track)
# ---------------------------------------------------------------------------
def validate_multifin_official(canonical_labels: set[str]) -> bool:
    print_section("2. MultiFin Multilingual Multi-Label Benchmark (Official Track)")
    mf_dir = REPO_ROOT / "data" / "processed" / "multifin"
    check_manifest(
        mf_dir, "clean_multifin_official", [f"{s}.jsonl" for s in ("train", "validation", "test")]
    )

    expected_counts = {"train": 3183, "validation": 796, "test": 995}
    expected_en = {"train": 1747, "validation": 437, "test": 546}
    expected_tr = {"train": 1436, "validation": 359, "test": 449}

    all_ids = set()
    split_ids: dict[str, set[str]] = {}
    split_fingerprints: dict[str, set[str]] = {}
    split_records: dict[str, list[dict]] = {}
    total_multi_label = 0
    passed = True

    for split_name, expected_total in expected_counts.items():
        file_path = mf_dir / f"{split_name}.jsonl"
        if not file_path.exists():
            print(f"[FAIL] Missing file: {file_path}")
            return False

        records = load_audit_records(file_path)
        split_records[split_name] = records
        split_ids[split_name] = {r["id"] for r in records}
        split_fingerprints[split_name] = {compute_content_fingerprint(r["text"]) for r in records}
        en_count = sum(1 for r in records if r["language"] == "en")
        tr_count = sum(1 for r in records if r["language"] == "tr")

        if (
            len(records) != expected_total
            or en_count != expected_en[split_name]
            or tr_count != expected_tr[split_name]
        ):
            print(
                f"[FAIL] Split '{split_name}' size mismatch: got {len(records)}, "
                f"expected {expected_total}"
            )
            passed = False
        else:
            print(
                f"  [OK] Split '{split_name}': {len(records)} records "
                f"({en_count} en, {tr_count} tr)"
            )

        for r in records:
            all_ids.add(r["id"])

            try:
                validate_record(r, allowed_labels=canonical_labels)
            except Exception as exc:
                rec_id = r.get("id", "unknown")
                print(f"[FAIL] MultiFin record {rec_id} schema validation failed: {exc}")
                passed = False

            if r.get("split") != split_name:
                print(
                    f"[FAIL] MultiFin record {r.get('id')} split mismatch: "
                    f"got '{r.get('split')}', expected '{split_name}'"
                )
                passed = False

            if len(r.get("labels", [])) > 1:
                total_multi_label += 1
            for lbl in r.get("labels", []):
                if lbl not in canonical_labels:
                    print(f"[FAIL] Record {r['id']} has unknown label: {lbl}")
                    passed = False

    expected_sum = sum(expected_counts.values())
    if len(all_ids) != expected_sum:
        print(f"[FAIL] Unique ID count mismatch: got {len(all_ids)}, expected {expected_sum}")
        passed = False
    else:
        print(f"  [OK] Unique IDs: {len(all_ids)} / {expected_sum} (zero duplicates)")

    t_v = split_ids["train"] & split_ids["validation"]
    t_te = split_ids["train"] & split_ids["test"]
    v_te = split_ids["validation"] & split_ids["test"]
    if t_v or t_te or v_te:
        print(
            f"[FAIL] Split ID leakage in MultiFin: "
            f"t_v={len(t_v)}, t_te={len(t_te)}, v_te={len(v_te)}"
        )
        passed = False
    else:
        print("  [OK] Split ID leakage: 0 (zero ID overlap between train, val, and test)")

    if total_multi_label != 1591:
        print(f"[FAIL] Multi-label count unexpected: got {total_multi_label}, expected 1591")
        passed = False
    else:
        pct = total_multi_label / sum(expected_counts.values()) * 100
        print(f"  [OK] Multi-label preservation: {total_multi_label} examples ({pct:.2f}%)")

    # Content-fingerprint leakage audit (official track documentation)
    train_fps = split_fingerprints["train"]
    val_leaks = sum(
        1
        for r in split_records["validation"]
        if compute_content_fingerprint(r["text"]) in train_fps
    )
    test_leaks = sum(
        1 for r in split_records["test"] if compute_content_fingerprint(r["text"]) in train_fps
    )
    val_fps = split_fingerprints["validation"]
    test_fps = split_fingerprints["test"]
    vt_overlap = len(val_fps & test_fps)
    if (val_leaks, test_leaks, vt_overlap) != (120, 157, 45):
        raise ValueError("Official MultiFin overlap differs from the pinned reference dataset.")

    print(
        f"  [INFO] Documented official MultiFin content-fingerprint leakage:\n"
        f"         Validation records leaking from train: {val_leaks}\n"
        f"         Test records leaking from train: {test_leaks}\n"
        f"         Validation/Test content overlap: {vt_overlap}\n"
        f"         (Official track preserved for baseline comparison; see leakage-free track)"
    )

    return passed


# ---------------------------------------------------------------------------
# Track 3: MultiFin Leakage-Free Evaluation Track
# ---------------------------------------------------------------------------
def validate_multifin_leakage_free(canonical_labels: set[str]) -> bool:
    print_section("3. MultiFin Leakage-Free Evaluation Track")
    lf_dir = REPO_ROOT / "data" / "splits" / "multifin" / "leakage_free"

    expected_files = ["train.jsonl", "validation.jsonl", "test.jsonl"]
    for fname in expected_files:
        fpath = lf_dir / fname
        if not fpath.exists():
            print(f"[FAIL] Missing file in leakage-free track: {fpath}")
            return False

    manifest = check_manifest(lf_dir, "multifin_leakage_free", expected_files)
    if (
        manifest["generation_parameters"].get("overlap_policy")
        != "preserve_test_remove_from_validation"
    ):
        raise ValueError("MultiFin evaluation manifest has an obsolete overlap policy.")

    train_records = load_audit_records(lf_dir / "train.jsonl")
    val_records = load_audit_records(lf_dir / "validation.jsonl")
    test_records = load_audit_records(lf_dir / "test.jsonl")

    passed = True

    # Validate schemas
    for r in train_records + val_records + test_records:
        try:
            validate_record(r, allowed_labels=canonical_labels)
        except Exception as exc:
            rec_id = r.get("id", "unknown")
            print(f"[FAIL] Leakage-free MultiFin record {rec_id} schema error: {exc}")
            passed = False

    # Check ID leakage
    train_ids = {r["id"] for r in train_records}
    val_ids = {r["id"] for r in val_records}
    test_ids = {r["id"] for r in test_records}

    if (train_ids & val_ids) or (train_ids & test_ids) or (val_ids & test_ids):
        print("[FAIL] ID leakage detected in MultiFin leakage-free track")
        passed = False
    else:
        print("  [OK] Split ID leakage: 0 (zero ID overlap)")

    # Check content-fingerprint leakage from training reference
    train_fps = {compute_content_fingerprint(r["text"]) for r in train_records}
    val_fps = {compute_content_fingerprint(r["text"]) for r in val_records}
    test_fps = {compute_content_fingerprint(r["text"]) for r in test_records}

    leaking_val = train_fps & val_fps
    leaking_test = train_fps & test_fps

    if leaking_val:
        print(f"[FAIL] Validation set still contains {len(leaking_val)} records leaking from train")
        passed = False
    else:
        print("  [OK] Train -> Validation content leakage: 0 (completely eliminated)")

    if leaking_test:
        print(f"[FAIL] Test set still contains {len(leaking_test)} records leaking from train")
        passed = False
    else:
        print("  [OK] Train -> Test content leakage: 0 (completely eliminated)")

    official_dir = REPO_ROOT / "data/processed/multifin"
    official = {
        s: load_audit_records(official_dir / f"{s}.jsonl") for s in ("train", "validation", "test")
    }
    expected_test_records = [
        r for r in official["test"] if compute_content_fingerprint(r["text"]) not in train_fps
    ]
    expected_test_fps = {compute_content_fingerprint(r["text"]) for r in expected_test_records}
    expected_val_records = [
        r
        for r in official["validation"]
        if compute_content_fingerprint(r["text"]) not in train_fps | expected_test_fps
    ]
    expected = {
        "train": official["train"],
        "validation": expected_val_records,
        "test": expected_test_records,
    }
    for split, records in zip(
        ("train", "validation", "test"), (train_records, val_records, test_records), strict=True
    ):
        expected_by_id = {r["id"]: {**r, "original_split": split} for r in expected[split]}
        if {r["id"]: r for r in records} != expected_by_id:
            raise ValueError(
                f"MultiFin {split} does not match the policy-filtered official records."
            )
    check_disjoint_partitions(
        {"train": train_records, "validation": val_records, "test": test_records}
    )
    expected_train = len(expected["train"])
    expected_val = len(expected["validation"])
    expected_test = len(expected["test"])
    params = manifest["generation_parameters"]
    val_train_leaks = sum(
        compute_content_fingerprint(r["text"]) in train_fps for r in official["validation"]
    )
    expected_stats = {
        "removed_val_leaks": val_train_leaks,
        "removed_val_test_overlap": len(official["validation"]) - val_train_leaks - expected_val,
        "removed_test_leaks": len(official["test"]) - expected_test,
        "val_test_overlap_count": 0,
    }
    if any(params.get(key) != value for key, value in expected_stats.items()):
        raise ValueError("MultiFin manifest exclusion statistics do not match the source records.")

    if len(train_records) != expected_train:
        print(f"[FAIL] Train count mismatch: got {len(train_records)}, expected {expected_train}")
        passed = False
    else:
        print(f"  [OK] Train count: {len(train_records)} (matches official reference)")

    if len(val_records) != expected_val:
        print(f"[FAIL] Validation count mismatch: got {len(val_records)}, expected {expected_val}")
        passed = False
    else:
        print(f"  [OK] Validation count: {len(val_records)} (train and test contamination removed)")

    if len(test_records) != expected_test:
        print(f"[FAIL] Test count mismatch: got {len(test_records)}, expected {expected_test}")
        passed = False
    else:
        print(f"  [OK] Test count: {len(test_records)} (157 leaks removed from 995)")

    vt_overlap = len(val_fps & test_fps)
    print(f"  [INFO] Validation/Test content overlap: {vt_overlap} content fingerprints")

    print("  [OK] Full traceability verified against original records and splits")

    return passed


# ---------------------------------------------------------------------------
# Track 4: Clean MN-DS Benchmark
# ---------------------------------------------------------------------------
def validate_mn_ds(canonical_labels: set[str]) -> bool:
    print_section("4. MN-DS Hierarchical News Benchmark (Clean)")
    mnds_dir = REPO_ROOT / "data" / "processed" / "mn_ds"
    check_manifest(mnds_dir, "clean_mn_ds", [f"{s}.jsonl" for s in ("train", "validation", "test")])

    expected_counts = {"train": 7344, "validation": 1574, "test": 1573}
    all_ids = set()
    split_ids: dict[str, set[str]] = {}
    split_fingerprints: dict[str, set[str]] = {}
    split_labels: dict[str, set[str]] = {}
    total_multi_label = 0
    passed = True

    for split_name, expected_total in expected_counts.items():
        file_path = mnds_dir / f"{split_name}.jsonl"
        if not file_path.exists():
            print(f"[FAIL] Missing file: {file_path}")
            return False

        records = load_audit_records(file_path)
        split_ids[split_name] = {r["id"] for r in records}
        split_fingerprints[split_name] = {compute_content_fingerprint(r["text"]) for r in records}
        split_labels[split_name] = {lbl for r in records for lbl in r.get("labels", [])}

        if len(records) != expected_total:
            print(
                f"[FAIL] Split '{split_name}' size mismatch: got {len(records)}, "
                f"expected {expected_total}"
            )
            passed = False
        else:
            num_cats = len(split_labels[split_name])
            print(
                f"  [OK] Split '{split_name}': {len(records)} records "
                f"(covers {num_cats} categories)"
            )

        for r in records:
            all_ids.add(r["id"])

            try:
                validate_record(r, allowed_labels=canonical_labels)
            except Exception as exc:
                rec_id = r.get("id", "unknown")
                print(f"[FAIL] MN-DS record {rec_id} schema validation failed: {exc}")
                passed = False

            if r.get("split") != split_name:
                print(
                    f"[FAIL] MN-DS record {r.get('id')} split mismatch: "
                    f"got '{r.get('split')}', expected '{split_name}'"
                )
                passed = False

            if len(r.get("labels", [])) > 1:
                total_multi_label += 1
            for lbl in r.get("labels", []):
                if lbl not in canonical_labels:
                    print(f"[FAIL] Record {r['id']} has unknown label: {lbl}")
                    passed = False

    if len(all_ids) != 10491:
        print(f"[FAIL] Unique ID count mismatch: got {len(all_ids)}, expected 10491")
        passed = False
    else:
        print(f"  [OK] Unique articles: {len(all_ids)} / 10,491 (zero duplicate article IDs)")

    # Split ID leakage
    t_v = split_ids["train"] & split_ids["validation"]
    t_te = split_ids["train"] & split_ids["test"]
    v_te = split_ids["validation"] & split_ids["test"]
    if t_v or t_te or v_te:
        print(
            f"[FAIL] Article ID leakage in MN-DS: "
            f"t_v={len(t_v)}, t_te={len(t_te)}, v_te={len(v_te)}"
        )
        passed = False
    else:
        print("  [OK] Article ID leakage: 0 (zero overlap between train, val, and test)")

    # Content-fingerprint leakage
    fp_tv = split_fingerprints["train"] & split_fingerprints["validation"]
    fp_tte = split_fingerprints["train"] & split_fingerprints["test"]
    fp_vte = split_fingerprints["validation"] & split_fingerprints["test"]
    if fp_tv or fp_tte or fp_vte:
        print(
            f"[FAIL] Content-fingerprint leakage in MN-DS: "
            f"t_v={len(fp_tv)}, t_te={len(fp_tte)}, v_te={len(fp_vte)}"
        )
        passed = False
    else:
        print("  [OK] Content-fingerprint leakage: 0 (zero content overlap after atomic grouping)")

    if total_multi_label != 392:
        print(f"[FAIL] Multi-label article count unexpected: got {total_multi_label}, expected 392")
        passed = False
    else:
        print(f"  [OK] Multi-label preservation: {total_multi_label} articles (3.74%)")

    all_covered = set().union(*split_labels.values())
    if (
        len(all_covered) != 109
        or len(split_labels["train"]) != 109
        or len(split_labels["validation"]) != 109
        or len(split_labels["test"]) != 109
    ):
        print(
            f"[FAIL] Category coverage incomplete: train={len(split_labels['train'])}, "
            f"val={len(split_labels['validation'])}, test={len(split_labels['test'])}"
        )
        passed = False
    else:
        print("  [OK] Full 109/109 category coverage achieved across train, val, and test splits")

    return passed


# ---------------------------------------------------------------------------
# Track 5: Noisy Robustness Benchmark
# ---------------------------------------------------------------------------
def validate_noisy(canonical_labels: set[str]) -> bool:
    print_section("5. Noisy Robustness Benchmark")
    noisy_base = REPO_ROOT / "data" / "noisy" / "combined" / "medium"

    check_manifest(
        noisy_base,
        "noisy_robustness_benchmark",
        [f"{d}/{s}.jsonl" for d in ("sib200", "multifin", "mn_ds") for s in ("validation", "test")],
    )

    configs = {
        "sib200": {
            "clean_dir": REPO_ROOT / "data" / "processed" / "sib200",
            "splits": {"validation": 198, "test": 408},
        },
        "multifin": {
            "clean_dir": REPO_ROOT / "data" / "processed" / "multifin",
            "splits": {"validation": 796, "test": 995},
        },
        "mn_ds": {
            "clean_dir": REPO_ROOT / "data" / "processed" / "mn_ds",
            "splits": {"validation": 1574, "test": 1573},
        },
    }

    passed = True

    for ds_name, cfg in configs.items():
        ds_dir = noisy_base / ds_name
        manifest = check_manifest(ds_dir, f"noisy_{ds_name}", ["validation.jsonl", "test.jsonl"])

        train_file = ds_dir / "train.jsonl"
        if train_file.exists():
            print(f"[FAIL] Noisy train file was accidentally generated: {train_file}")
            passed = False

        clean_train_ids = set()
        clean_train_path = cfg["clean_dir"] / "train.jsonl"
        if clean_train_path.exists():
            clean_train_ids = {r["id"] for r in load_audit_records(clean_train_path)}

        for split_name, expected_count in cfg["splits"].items():
            noisy_file = ds_dir / f"{split_name}.jsonl"
            clean_file = cfg["clean_dir"] / f"{split_name}.jsonl"

            if not noisy_file.exists():
                print(f"[FAIL] Missing noisy file: {noisy_file}")
                return False
            if not clean_file.exists():
                print(f"[FAIL] Missing clean source file: {clean_file}")
                return False

            clean_records = {r["id"]: r for r in load_audit_records(clean_file)}
            noisy_records = load_audit_records(noisy_file)
            originals = [r.get("original_id") for r in noisy_records]
            if len(originals) != len(set(originals)) or set(originals) != set(clean_records):
                raise ValueError(
                    f"{ds_name} noisy {split_name} must cover each clean record exactly once."
                )

            if len(noisy_records) != expected_count:
                print(
                    f"[FAIL] {ds_name} noisy {split_name} count mismatch: "
                    f"got {len(noisy_records)}, expected {expected_count}"
                )
                passed = False
            else:
                print(f"  [OK] {ds_name} noisy {split_name}: {len(noisy_records)} records")

            for nr in noisy_records:
                try:
                    validate_record(nr, allowed_labels=canonical_labels)
                except Exception as exc:
                    rec_id = nr.get("id", "unknown")
                    print(f"[FAIL] Noisy record {rec_id} schema validation failed: {exc}")
                    passed = False

                orig_id = nr.get("original_id")
                if not orig_id or orig_id not in clean_records:
                    print(f"[FAIL] Noisy record {nr.get('id')} has invalid original_id: {orig_id}")
                    passed = False
                    continue

                if orig_id in clean_train_ids:
                    print(f"[FAIL] Noisy record {nr.get('id')} mapped to clean train {orig_id}")
                    passed = False

                clean_r = clean_records[orig_id]
                params = manifest["generation_parameters"]
                expected = create_noisy_record(
                    clean_r,
                    strategy=params["strategy"],
                    severity=params["severity"],
                    base_seed=params["base_seed"],
                )
                if nr != expected:
                    raise ValueError(
                        f"Noisy record {nr['id']} does not reproduce from its source and seed."
                    )

                for field in ["labels", "language", "source", "license", "split"]:
                    if nr.get(field) != clean_r.get(field):
                        print(f"[FAIL] Noisy record {nr['id']} changed invariant '{field}'")
                        passed = False

                if "pair_id" in clean_r and nr.get("pair_id") != clean_r["pair_id"]:
                    print(
                        f"[FAIL] Noisy record {nr['id']} changed pair_id: "
                        f"{nr.get('pair_id')} vs {clean_r['pair_id']}"
                    )
                    passed = False

                if not nr.get("text", "").strip():
                    print(f"[FAIL] Noisy record {nr['id']} has empty text")
                    passed = False
                elif nr["text"] == clean_r["text"]:
                    print(f"[FAIL] Noisy record {nr['id']} text was not changed from clean source")
                    passed = False

                if "base_seed" not in nr:
                    print(f"[FAIL] Noisy record {nr['id']} missing base_seed metadata")
                    passed = False

    return passed


# ---------------------------------------------------------------------------
# Track 6: Synthetic EN/TR Code-Switch Benchmark
# ---------------------------------------------------------------------------
def validate_code_switch(canonical_labels: set[str]) -> bool:
    print_section("6. Synthetic EN/TR Code-Switch Benchmark")
    cs_dir = REPO_ROOT / "data" / "processed" / "code_switch" / "chunk_mix" / "balanced"

    manifest = check_manifest(cs_dir, "code_switch_sib200", ["validation.jsonl", "test.jsonl"])

    train_file = cs_dir / "train.jsonl"
    if train_file.exists():
        print(f"[FAIL] Code-switch train file was accidentally generated: {train_file}")
        return False

    expected_splits = {"validation": 99, "test": 204}
    sib_dir = REPO_ROOT / "data" / "processed" / "sib200"

    all_ids = set()
    passed = True

    for split_name, expected_count in expected_splits.items():
        cs_file = cs_dir / f"{split_name}.jsonl"
        sib_file = sib_dir / f"{split_name}.jsonl"

        if not cs_file.exists():
            print(f"[FAIL] Missing code-switch file: {cs_file}")
            return False
        if not sib_file.exists():
            print(f"[FAIL] Missing clean source SIB-200 file: {sib_file}")
            return False

        sib_records = load_audit_records(sib_file)
        sib_by_id = {r["id"]: r for r in sib_records}
        cs_records = load_audit_records(cs_file)
        expected_pairs = pair_aligned_records(sib_records)
        params = manifest["generation_parameters"]
        expected_records = [
            create_code_switched_record(
                en,
                tr,
                strategy=params["strategy"],
                strength=params["strength"],
                base_seed=params["base_seed"],
            )
            for en, tr in expected_pairs
        ]
        if {r["id"]: r for r in cs_records} != {r["id"]: r for r in expected_records}:
            raise ValueError(
                f"Code-switch {split_name} does not reproduce from all aligned source pairs."
            )

        if len(cs_records) != expected_count:
            print(
                f"[FAIL] Code-switch {split_name} count mismatch: "
                f"got {len(cs_records)}, expected {expected_count}"
            )
            passed = False
        else:
            print(f"  [OK] Code-switch {split_name}: {len(cs_records)} records")

        for r in cs_records:
            all_ids.add(r["id"])

            try:
                validate_record(r, allowed_labels=canonical_labels)
            except Exception as exc:
                rec_id = r.get("id", "unknown")
                print(f"[FAIL] Code-switch record {rec_id} schema validation failed: {exc}")
                passed = False

            if r.get("language") != "en-tr":
                print(
                    f"[FAIL] Code-switch record {r.get('id')} language is "
                    f"'{r.get('language')}', expected 'en-tr'"
                )
                passed = False

            en_id = r.get("en_id")
            tr_id = r.get("tr_id")
            if not en_id or en_id not in sib_by_id:
                print(f"[FAIL] Code-switch record {r.get('id')} has invalid en_id: {en_id}")
                passed = False
                continue
            if not tr_id or tr_id not in sib_by_id:
                print(f"[FAIL] Code-switch record {r.get('id')} has invalid tr_id: {tr_id}")
                passed = False
                continue

            en_r = sib_by_id[en_id]
            tr_r = sib_by_id[tr_id]

            if en_r.get("pair_id") != tr_r.get("pair_id"):
                print(
                    f"[FAIL] Code-switch record {r.get('id')} en_id and tr_id "
                    f"belong to different pair_ids"
                )
                passed = False

            if r.get("pair_id") != en_r.get("pair_id"):
                print(f"[FAIL] Code-switch record {r.get('id')} pair_id does not match source")
                passed = False

            if en_r.get("labels") != tr_r.get("labels") or r.get("labels") != en_r.get("labels"):
                print(f"[FAIL] Code-switch record {r.get('id')} labels do not match source")
                passed = False

            if r.get("split") != split_name:
                print(
                    f"[FAIL] Code-switch record {r.get('id')} split mismatch: "
                    f"got '{r.get('split')}', expected '{split_name}'"
                )
                passed = False

            if not r.get("text", "").strip():
                print(f"[FAIL] Code-switch record {r.get('id')} has empty text")
                passed = False

            if "base_seed" not in r:
                print(f"[FAIL] Code-switch record {r.get('id')} missing base_seed")
                passed = False

    expected_sum = sum(expected_splits.values())
    if len(all_ids) != expected_sum:
        print(f"[FAIL] Code-switch IDs not unique: got {len(all_ids)}, expected {expected_sum}")
        passed = False
    else:
        print(f"  [OK] Unique code-switched IDs: {len(all_ids)} / {expected_sum}")

    return passed


# ---------------------------------------------------------------------------
# Track 7: Near-OOD SIB-200 Leave-One-Topic-Out Benchmark (7 Folds)
# ---------------------------------------------------------------------------
def validate_near_ood(canonical_labels: set[str]) -> bool:
    print_section("7. Near-OOD SIB-200 Leave-One-Topic-Out Benchmark (7 Folds)")
    near_dir = REPO_ROOT / "data" / "ood" / "near" / "sib200"

    expected_topics = [
        "arts_culture_entertainment_media",
        "geography",
        "health",
        "politics",
        "science_technology",
        "sport",
        "travel",
    ]

    expected_files = [
        "train_id.jsonl",
        "validation_id.jsonl",
        "validation_ood.jsonl",
        "test_id.jsonl",
        "test_ood.jsonl",
    ]

    passed = True

    for topic in expected_topics:
        fold_dir = near_dir / topic
        if not fold_dir.exists():
            print(f"[FAIL] Missing fold directory: {fold_dir}")
            return False

        check_manifest(fold_dir, f"near_ood_sib200_{topic}", expected_files)

        fold_files = {}
        for fname in expected_files:
            fpath = fold_dir / fname
            if not fpath.exists():
                print(f"[FAIL] Missing file in fold {topic}: {fpath}")
                return False
            fold_files[fname] = load_audit_records(fpath)

        train_id_records = fold_files["train_id.jsonl"]
        val_id_records = fold_files["validation_id.jsonl"]
        val_ood_records = fold_files["validation_ood.jsonl"]
        test_id_records = fold_files["test_id.jsonl"]
        test_ood_records = fold_files["test_ood.jsonl"]
        partitions = {
            "train": train_id_records,
            "validation": val_id_records + val_ood_records,
            "test": test_id_records + test_ood_records,
        }
        check_disjoint_partitions(partitions)
        for split, records in partitions.items():
            pair_aligned_records(records)
            source_records = load_audit_records(
                REPO_ROOT / "data/processed/sib200" / f"{split}.jsonl"
            )
            id_expected = {r["id"]: r for r in source_records if topic not in r["labels"]}
            id_actual = {r["id"]: r for r in records if not r.get("is_ood")}
            if id_actual != id_expected:
                raise ValueError(
                    f"Fold {topic} {split}: ID records differ from the six-topic clean source."
                )
            ood_expected = {
                (r["pair_id"], r["language"]): r
                for r in source_records
                if topic in r["labels"] and split != "train"
            }
            ood_actual = {(r["pair_id"], r["language"]): r for r in records if r.get("is_ood")}
            if set(ood_actual) != set(ood_expected):
                raise ValueError(f"Fold {topic} {split}: held-out source coverage mismatch.")
            for key, r in ood_actual.items():
                orig = ood_expected[key]
                if (
                    any(
                        r.get(field) != orig.get(field)
                        for field in (
                            "text",
                            "language",
                            "source",
                            "license",
                            "split",
                            "pair_id",
                            "source_id",
                        )
                    )
                    or r["ood_type"] != "near_ood"
                    or r["source_label"] != topic
                ):
                    raise ValueError(
                        f"Fold {topic} OOD record {r['id']} changed source metadata or text."
                    )

        for r in train_id_records + val_id_records + test_id_records:
            try:
                validate_record(r, allowed_labels=canonical_labels)
            except Exception as exc:
                print(f"[FAIL] Fold {topic} ID record {r.get('id')} failed schema: {exc}")
                passed = False

        for r in val_ood_records:
            try:
                validate_ood_record(r)
            except Exception as exc:
                print(f"[FAIL] Fold {topic} val_ood record {r.get('id')} failed OOD schema: {exc}")
                passed = False
            if r.get("split") != "validation":
                print(
                    f"[FAIL] Fold {topic} val_ood {r.get('id')} split is "
                    f"'{r.get('split')}', expected 'validation'"
                )
                passed = False
            if r.get("source_label") != topic:
                print(
                    f"[FAIL] Fold {topic} val_ood {r.get('id')} source_label "
                    f"mismatch: {r.get('source_label')}"
                )
                passed = False

        for r in test_ood_records:
            try:
                validate_ood_record(r)
            except Exception as exc:
                print(f"[FAIL] Fold {topic} test_ood record {r.get('id')} failed OOD schema: {exc}")
                passed = False
            if r.get("split") != "test":
                print(
                    f"[FAIL] Fold {topic} test_ood {r.get('id')} split is "
                    f"'{r.get('split')}', expected 'test'"
                )
                passed = False
            if r.get("source_label") != topic:
                print(
                    f"[FAIL] Fold {topic} test_ood {r.get('id')} source_label "
                    f"mismatch: {r.get('source_label')}"
                )
                passed = False

        train_labels = {lbl for r in train_id_records for lbl in r.get("labels", [])}
        if topic in train_labels:
            print(f"[FAIL] Fold {topic}: held-out topic '{topic}' found in train_id.jsonl!")
            passed = False

        remaining_expected = set(expected_topics) - {topic}
        if train_labels != remaining_expected:
            print(
                f"[FAIL] Fold {topic}: train_id labels {train_labels} != "
                f"expected {remaining_expected}"
            )
            passed = False

        train_pairs = defaultdict(list)
        for r in train_id_records:
            train_pairs[r["pair_id"]].append(r)
        for p_id, p_recs in train_pairs.items():
            if len(p_recs) != 2 or {r["language"] for r in p_recs} != {"en", "tr"}:
                print(f"[FAIL] Fold {topic} pair {p_id} in train_id not exact EN/TR pair")
                passed = False

        train_pids = set(train_pairs.keys())
        val_ood_pids = {r["pair_id"] for r in val_ood_records}
        test_ood_pids = {r["pair_id"] for r in test_ood_records}

        if (train_pids & val_ood_pids) or (train_pids & test_ood_pids):
            print(f"[FAIL] Fold {topic}: training pair leaked into OOD val/test!")
            passed = False

        t_cnt = len(train_id_records)
        v_cnt = len(val_ood_records)
        te_cnt = len(test_ood_records)
        print(f"  [OK] Fold '{topic}': train_id={t_cnt}, val_ood={v_cnt}, test_ood={te_cnt}")

    return passed


# ---------------------------------------------------------------------------
# Track 8: Far-OOD Benchmark (MASSIVE)
# ---------------------------------------------------------------------------
def validate_far_ood() -> bool:
    print_section("8. Far-OOD Benchmark (Amazon MASSIVE EN/TR)")
    far_dir = REPO_ROOT / "data" / "ood" / "far" / "massive"

    check_manifest(far_dir, "far_ood_massive", ["validation_ood.jsonl", "test_ood.jsonl"])

    val_file = far_dir / "validation_ood.jsonl"
    test_file = far_dir / "test_ood.jsonl"

    if not val_file.exists():
        print(f"[FAIL] Missing Far-OOD file: {val_file}")
        return False
    if not test_file.exists():
        print(f"[FAIL] Missing Far-OOD file: {test_file}")
        return False

    val_records = load_audit_records(val_file)
    test_records = load_audit_records(test_file)
    check_disjoint_partitions({"validation": val_records, "test": test_records})
    raw_dir = REPO_ROOT / "data/raw/massive"
    raw_by_language = {
        lang: {str(r["id"]): r for r in load_jsonl(raw_dir / f"{locale}.jsonl")}
        for lang, locale in (("en", "en-US"), ("tr", "tr-TR"))
    }
    for r in val_records + test_records:
        raw = raw_by_language[r["language"]].get(str(r["source_id"]))
        if raw is None or raw["partition"] != {"validation": "dev", "test": "test"}[r["split"]]:
            raise ValueError(
                f"Far-OOD {r['id']} does not come from the declared upstream partition."
            )
        if (
            raw["scenario"] not in MASSIVE_ALLOWED_SCENARIOS
            or r["source_label"] != f"{raw['scenario']}:{raw['intent']}"
            or r["text"] != raw["utt"].strip()
            or r["ood_type"] != "far_ood"
        ):
            raise ValueError(
                f"Far-OOD {r['id']} changed source content or uses an excluded scenario."
            )

    passed = True

    expected_val = 500
    expected_test = 1000

    if len(val_records) != expected_val:
        print(
            f"[FAIL] Far-OOD validation count mismatch: got {len(val_records)}, "
            f"expected {expected_val}"
        )
        passed = False
    else:
        print(f"  [OK] Far-OOD validation: {len(val_records)} records")

    if len(test_records) != expected_test:
        print(
            f"[FAIL] Far-OOD test count mismatch: got {len(test_records)}, "
            f"expected {expected_test}"
        )
        passed = False
    else:
        print(f"  [OK] Far-OOD test: {len(test_records)} records")

    for r in val_records + test_records:
        try:
            validate_ood_record(r)
        except Exception as exc:
            print(f"[FAIL] Far-OOD record {r.get('id', 'unknown')} failed OOD schema: {exc}")
            passed = False

    for split_name, recs in [("validation", val_records), ("test", test_records)]:
        pairs = defaultdict(list)
        for r in recs:
            pairs[r["pair_id"]].append(r)

        for p_id, p_recs in pairs.items():
            if len(p_recs) != 2:
                print(f"[FAIL] Far-OOD {split_name} pair {p_id} has {len(p_recs)} records")
                passed = False
                continue

            langs = {r["language"] for r in p_recs}
            if langs != {"en", "tr"}:
                print(f"[FAIL] Far-OOD {split_name} pair {p_id} languages {langs} != {'en', 'tr'}")
                passed = False
                continue

            r_en = next(r for r in p_recs if r["language"] == "en")
            r_tr = next(r for r in p_recs if r["language"] == "tr")

            if r_en.get("source_label") != r_tr.get("source_label"):
                print(f"[FAIL] Far-OOD pair {p_id} source_label mismatch")
                passed = False

    val_pids = {r["pair_id"] for r in val_records}
    test_pids = {r["pair_id"] for r in test_records}
    if val_pids & test_pids:
        print(
            f"[FAIL] Far-OOD pair_id overlap between val and test: " f"{len(val_pids & test_pids)}"
        )
        passed = False
    else:
        print("  [OK] Validation and test pair IDs are completely disjoint (zero overlap)")

    return passed


# ---------------------------------------------------------------------------
# Main Audit Entrypoint
# ---------------------------------------------------------------------------
def main() -> int:
    tax_path = REPO_ROOT / "taxonomy" / "taxonomy.json"
    canonical_labels = load_canonical_label_ids(tax_path)
    print(f"Loaded canonical taxonomy with {len(canonical_labels)} labels.")

    audits = {
        "Clean SIB-200": lambda: validate_sib200(canonical_labels),
        "Clean MultiFin (Official)": lambda: validate_multifin_official(canonical_labels),
        "MultiFin Leakage-Free Track": lambda: validate_multifin_leakage_free(canonical_labels),
        "Clean MN-DS": lambda: validate_mn_ds(canonical_labels),
        "Noisy Robustness Benchmark": lambda: validate_noisy(canonical_labels),
        "Synthetic EN/TR Code-Switch Benchmark": lambda: validate_code_switch(canonical_labels),
        "Near-OOD (7 Folds)": lambda: validate_near_ood(canonical_labels),
        "Far-OOD (MASSIVE)": validate_far_ood,
    }
    results = {}
    for track, audit in audits.items():
        try:
            results[track] = audit()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"[FAIL] {track}: {exc}")
            results[track] = False

    print_section("AUDIT SUMMARY")
    all_passed = True
    for track_name, status in results.items():
        tag = "[PASS]" if status else "[FAIL]"
        print(f"  {tag} {track_name}")
        if not status:
            all_passed = False

    if all_passed:
        print("\n[PASSED] ALL 8 BENCHMARK TRACK AUDITS PASSED SUCCESSFULLY!\n")
        print(
            "Official MultiFin and its noisy variants retain documented overlap; "
            "use the leakage-free track for model selection and final evaluation."
        )
        return 0
    else:
        print("\n[FAIL] ONE OR MORE BENCHMARK TRACK AUDITS FAILED. See log above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
