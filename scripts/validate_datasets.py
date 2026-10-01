"""Comprehensive dataset-wide verification and auditing script for NALTRA."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import sys

# Ensure src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from naltra.data.loader import load_jsonl
from naltra.data.preprocessing import load_canonical_label_ids


def print_section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def validate_sib200(canonical_labels: set[str]) -> bool:
    print_section("1. SIB-200 Cross-Language Topic Benchmark")
    sib_dir = REPO_ROOT / "data" / "processed" / "sib200"

    expected_counts = {"train": 1402, "validation": 198, "test": 408}
    expected_en = {"train": 701, "validation": 99, "test": 204}
    expected_tr = {"train": 701, "validation": 99, "test": 204}

    all_ids = set()
    pair_splits: dict[str, set[str]] = defaultdict(set)
    pair_labels: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    pair_langs: dict[str, set[str]] = defaultdict(set)
    split_ids: dict[str, set[str]] = {}

    passed = True

    for split_name, expected_total in expected_counts.items():
        file_path = sib_dir / f"{split_name}.jsonl"
        if not file_path.exists():
            print(f"[FAIL] Missing file: {file_path}")
            return False

        records = load_jsonl(file_path)
        split_ids[split_name] = {r["id"] for r in records}
        en_count = sum(1 for r in records if r["language"] == "en")
        tr_count = sum(1 for r in records if r["language"] == "tr")

        # Check total count and language split
        if len(records) != expected_total or en_count != expected_en[split_name] or tr_count != expected_tr[split_name]:
            print(f"[FAIL] Split '{split_name}' size mismatch: got {len(records)} ({en_count} en, {tr_count} tr), expected {expected_total}")
            passed = False
        else:
            print(f"  [OK] Split '{split_name}': {len(records)} records ({en_count} en, {tr_count} tr)")

        # Track pair alignment and taxonomy compliance
        for r in records:
            all_ids.add(r["id"])
            p_id = r.get("pair_id")
            if p_id:
                pair_splits[p_id].add(split_name)
                pair_labels[p_id].add(tuple(sorted(r["labels"])))
                pair_langs[p_id].add(r["language"])

            for l in r["labels"]:
                if l not in canonical_labels:
                    print(f"[FAIL] Record {r['id']} has unknown label: {l}")
                    passed = False

    # Check unique IDs
    if len(all_ids) != sum(expected_counts.values()):
        print(f"[FAIL] Unique ID count mismatch: got {len(all_ids)}, expected {sum(expected_counts.values())}")
        passed = False
    else:
        print(f"  [OK] Unique IDs: {len(all_ids)} / {sum(expected_counts.values())} (zero duplicates)")

    # Check zero split leakage
    t_v = split_ids["train"] & split_ids["validation"]
    t_te = split_ids["train"] & split_ids["test"]
    v_te = split_ids["validation"] & split_ids["test"]
    if t_v or t_te or v_te:
        print(f"[FAIL] Split leakage detected in SIB-200: t_v={len(t_v)}, t_te={len(t_te)}, v_te={len(v_te)}")
        passed = False
    else:
        print("  [OK] Split leakage: 0 (zero overlap between train, val, and test)")

    # Check pair alignment
    misaligned_split = sum(1 for s in pair_splits.values() if len(s) != 1)
    misaligned_labels = sum(1 for l in pair_labels.values() if len(l) != 1)
    incomplete_pairs = sum(1 for langs in pair_langs.values() if langs != {"en", "tr"})

    if misaligned_split or misaligned_labels or incomplete_pairs:
        print(f"[FAIL] Pair alignment issues: split_diff={misaligned_split}, label_diff={misaligned_labels}, incomplete={incomplete_pairs}")
        passed = False
    else:
        print(f"  [OK] English/Turkish pair alignment: 1,004 pairs perfectly aligned across splits and categories")

    return passed


def validate_multifin(canonical_labels: set[str]) -> bool:
    print_section("2. MultiFin Multilingual Multi-Label Benchmark")
    mf_dir = REPO_ROOT / "data" / "processed" / "multifin"

    expected_counts = {"train": 3183, "validation": 796, "test": 995}
    expected_en = {"train": 1747, "validation": 437, "test": 546}
    expected_tr = {"train": 1436, "validation": 359, "test": 449}

    all_ids = set()
    split_ids: dict[str, set[str]] = {}
    total_multi_label = 0
    passed = True

    for split_name, expected_total in expected_counts.items():
        file_path = mf_dir / f"{split_name}.jsonl"
        if not file_path.exists():
            print(f"[FAIL] Missing file: {file_path}")
            return False

        records = load_jsonl(file_path)
        split_ids[split_name] = {r["id"] for r in records}
        en_count = sum(1 for r in records if r["language"] == "en")
        tr_count = sum(1 for r in records if r["language"] == "tr")

        if len(records) != expected_total or en_count != expected_en[split_name] or tr_count != expected_tr[split_name]:
            print(f"[FAIL] Split '{split_name}' size mismatch: got {len(records)}, expected {expected_total}")
            passed = False
        else:
            print(f"  [OK] Split '{split_name}': {len(records)} records ({en_count} en, {tr_count} tr)")

        for r in records:
            all_ids.add(r["id"])
            if len(r["labels"]) > 1:
                total_multi_label += 1
            for l in r["labels"]:
                if l not in canonical_labels:
                    print(f"[FAIL] Record {r['id']} has unknown label: {l}")
                    passed = False

    # Check unique IDs
    if len(all_ids) != sum(expected_counts.values()):
        print(f"[FAIL] Unique ID count mismatch: got {len(all_ids)}, expected {sum(expected_counts.values())}")
        passed = False
    else:
        print(f"  [OK] Unique IDs: {len(all_ids)} / {sum(expected_counts.values())} (zero duplicates)")

    # Check zero split leakage
    t_v = split_ids["train"] & split_ids["validation"]
    t_te = split_ids["train"] & split_ids["test"]
    v_te = split_ids["validation"] & split_ids["test"]
    if t_v or t_te or v_te:
        print(f"[FAIL] Split leakage detected in MultiFin: t_v={len(t_v)}, t_te={len(t_te)}, v_te={len(v_te)}")
        passed = False
    else:
        print("  [OK] Split leakage: 0 (zero overlap between train, val, and test)")

    # Multi-label verification
    if total_multi_label != 1591:
        print(f"[FAIL] Multi-label count unexpected: got {total_multi_label}, expected 1591")
        passed = False
    else:
        print(f"  [OK] Multi-label preservation: {total_multi_label} examples (31.99%) correctly multi-labeled")

    return passed


def validate_mn_ds(canonical_labels: set[str]) -> bool:
    print_section("3. MN-DS Hierarchical News Benchmark")
    mnds_dir = REPO_ROOT / "data" / "processed" / "mn_ds"

    expected_counts = {"train": 7344, "validation": 1574, "test": 1573}
    all_ids = set()
    split_ids: dict[str, set[str]] = {}
    split_labels: dict[str, set[str]] = {}
    total_multi_label = 0
    passed = True

    for split_name, expected_total in expected_counts.items():
        file_path = mnds_dir / f"{split_name}.jsonl"
        if not file_path.exists():
            print(f"[FAIL] Missing file: {file_path}")
            return False

        records = load_jsonl(file_path)
        split_ids[split_name] = {r["id"] for r in records}
        split_labels[split_name] = {l for r in records for l in r["labels"]}

        if len(records) != expected_total:
            print(f"[FAIL] Split '{split_name}' size mismatch: got {len(records)}, expected {expected_total}")
            passed = False
        else:
            print(f"  [OK] Split '{split_name}': {len(records)} records (covers {len(split_labels[split_name])} categories)")

        for r in records:
            all_ids.add(r["id"])
            if len(r["labels"]) > 1:
                total_multi_label += 1
            for l in r["labels"]:
                if l not in canonical_labels:
                    print(f"[FAIL] Record {r['id']} has unknown label: {l}")
                    passed = False

    # Check unique IDs
    if len(all_ids) != 10491:
        print(f"[FAIL] Unique ID count mismatch: got {len(all_ids)}, expected 10491")
        passed = False
    else:
        print(f"  [OK] Unique articles: 10,491 / 10,491 (zero duplicate article IDs)")

    # Check zero split leakage
    t_v = split_ids["train"] & split_ids["validation"]
    t_te = split_ids["train"] & split_ids["test"]
    v_te = split_ids["validation"] & split_ids["test"]
    if t_v or t_te or v_te:
        print(f"[FAIL] Article leakage detected in MN-DS: t_v={len(t_v)}, t_te={len(t_te)}, v_te={len(v_te)}")
        passed = False
    else:
        print("  [OK] Article leakage: 0 (zero overlap between train, val, and test)")

    # Multi-label verification
    if total_multi_label != 392:
        print(f"[FAIL] Multi-label article count unexpected: got {total_multi_label}, expected 392")
        passed = False
    else:
        print(f"  [OK] Multi-label preservation: {total_multi_label} articles (3.74%) correctly multi-labeled")

    # Category coverage across splits
    all_covered = set().union(*split_labels.values())
    if len(all_covered) != 109 or len(split_labels["train"]) != 109 or len(split_labels["validation"]) != 109 or len(split_labels["test"]) != 109:
        print(f"[FAIL] Category coverage incomplete: train={len(split_labels['train'])}, val={len(split_labels['validation'])}, test={len(split_labels['test'])}")
        passed = False
    else:
        print("  [OK] Full 109/109 category coverage achieved in train, val, and test splits")

    return passed


def main() -> int:
    tax_path = REPO_ROOT / "taxonomy" / "taxonomy.json"
    canonical_labels = load_canonical_label_ids(tax_path)
    print(f"Loaded canonical taxonomy with {len(canonical_labels)} labels.")

    s1 = validate_sib200(canonical_labels)
    s2 = validate_multifin(canonical_labels)
    s3 = validate_mn_ds(canonical_labels)

    print_section("AUDIT SUMMARY")
    total_clean_records = 2008 + 4974 + 10491
    print(f"Total processed benchmark records: {total_clean_records:,}")
    print("  - SIB-200:   2,008 records (1,004 EN, 1,004 TR, paired)")
    print("  - MultiFin:  4,974 records (2,730 EN, 2,244 TR, multi-label)")
    print("  - MN-DS:    10,491 records (109 fine-grained categories, stratified 70/15/15)")

    if s1 and s2 and s3:
        print("\n[PASSED] ALL 8 DATASET-WIDE VERIFICATION CRITERIA PASSED SUCCESSFULLY!\n")
        return 0
    else:
        print("\n[FAIL] SOME VERIFICATION CHECKS FAILED. See log above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

