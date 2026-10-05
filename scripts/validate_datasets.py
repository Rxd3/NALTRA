"""Comprehensive dataset-wide verification and auditing script for NALTRA."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

# Ensure src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from naltra.data.loader import load_jsonl  # noqa: E402
from naltra.data.ood import validate_ood_record  # noqa: E402
from naltra.data.preprocessing import (  # noqa: E402
    compute_content_fingerprint,
    load_canonical_label_ids,
    validate_record,
)


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

        records = load_jsonl(file_path)
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
# Track 2: Noisy Robustness Benchmark
# ---------------------------------------------------------------------------
def validate_noisy(canonical_labels: set[str]) -> bool:
    print_section("2. Noisy Robustness Benchmark")
    noisy_base = REPO_ROOT / "data" / "noisy" / "combined" / "medium"

    manifest_path = noisy_base / "manifest.json"
    if not manifest_path.exists() and not (noisy_base / "sib200" / "manifest.json").exists():
        print(f"[FAIL] Missing manifest in noisy benchmark: {manifest_path}")
        return False
    print("  [OK] Manifest present and verified")

    configs = {
        "sib200": {
            "clean_dir": REPO_ROOT / "data" / "processed" / "sib200",
            "splits": {"validation": 198, "test": 408},
        },
    }

    passed = True

    for ds_name, cfg in configs.items():
        ds_dir = noisy_base / ds_name

        train_file = ds_dir / "train.jsonl"
        if train_file.exists():
            print(f"[FAIL] Noisy train file was accidentally generated: {train_file}")
            passed = False

        clean_train_ids = set()
        clean_train_path = cfg["clean_dir"] / "train.jsonl"
        if clean_train_path.exists():
            clean_train_ids = {r["id"] for r in load_jsonl(clean_train_path)}

        for split_name, expected_count in cfg["splits"].items():
            noisy_file = ds_dir / f"{split_name}.jsonl"
            clean_file = cfg["clean_dir"] / f"{split_name}.jsonl"

            if not noisy_file.exists():
                print(f"[FAIL] Missing noisy file: {noisy_file}")
                return False
            if not clean_file.exists():
                print(f"[FAIL] Missing clean source file: {clean_file}")
                return False

            clean_records = {r["id"]: r for r in load_jsonl(clean_file)}
            noisy_records = load_jsonl(noisy_file)

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
# Track 3: Synthetic EN/TR Code-Switch Benchmark
# ---------------------------------------------------------------------------
def validate_code_switch(canonical_labels: set[str]) -> bool:
    print_section("3. Synthetic EN/TR Code-Switch Benchmark")
    cs_dir = REPO_ROOT / "data" / "processed" / "code_switch" / "chunk_mix" / "balanced"

    manifest_path = cs_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[FAIL] Missing manifest in code-switch benchmark: {manifest_path}")
        return False
    print("  [OK] Manifest present and verified")

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

        sib_records = load_jsonl(sib_file)
        sib_by_id = {r["id"]: r for r in sib_records}
        cs_records = load_jsonl(cs_file)

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
# Track 4: Near-OOD SIB-200 Leave-One-Topic-Out Benchmark (7 Folds)
# ---------------------------------------------------------------------------
def validate_near_ood(canonical_labels: set[str]) -> bool:
    print_section("4. Near-OOD SIB-200 Leave-One-Topic-Out Benchmark (7 Folds)")
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

        manifest_path = fold_dir / "manifest.json"
        if not manifest_path.exists():
            print(f"[FAIL] Missing manifest in fold {topic}: {manifest_path}")
            return False

        fold_files = {}
        for fname in expected_files:
            fpath = fold_dir / fname
            if not fpath.exists():
                print(f"[FAIL] Missing file in fold {topic}: {fpath}")
                return False
            fold_files[fname] = load_jsonl(fpath)

        train_id_records = fold_files["train_id.jsonl"]
        val_id_records = fold_files["validation_id.jsonl"]
        val_ood_records = fold_files["validation_ood.jsonl"]
        test_id_records = fold_files["test_id.jsonl"]
        test_ood_records = fold_files["test_ood.jsonl"]

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
# Main Audit Entrypoint
# ---------------------------------------------------------------------------
def main() -> int:
    tax_path = REPO_ROOT / "taxonomy" / "taxonomy.json"
    canonical_labels = load_canonical_label_ids(tax_path)
    print(f"Loaded canonical taxonomy with {len(canonical_labels)} labels.")

    results = {
        "Clean SIB-200": validate_sib200(canonical_labels),
        "Noisy Robustness Benchmark": validate_noisy(canonical_labels),
        "Synthetic EN/TR Code-Switch Benchmark": validate_code_switch(canonical_labels),
        "Near-OOD (7 Folds)": validate_near_ood(canonical_labels),
    }

    print_section("AUDIT SUMMARY")
    all_passed = True
    for track_name, status in results.items():
        tag = "[PASS]" if status else "[FAIL]"
        print(f"  {tag} {track_name}")
        if not status:
            all_passed = False

    if all_passed:
        print("\n[PASSED] ALL SIB-200 BENCHMARK AUDITS PASSED SUCCESSFULLY!\n")
        return 0
    else:
        print("\n[FAIL] ONE OR MORE BENCHMARK TRACK AUDITS FAILED. See log above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
