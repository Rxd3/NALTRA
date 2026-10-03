from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Ensure src/ is on sys.path so naltra can be imported directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from naltra.data.code_switching import (  # noqa: E402
    create_code_switched_record,
    generate_code_switch_benchmarks,
    pair_aligned_records,
)
from naltra.data.loader import load_jsonl  # noqa: E402
from naltra.data.noise import create_noisy_record, generate_noisy_benchmarks  # noqa: E402
from naltra.data.ood import (  # noqa: E402
    generate_far_ood_benchmarks,
    generate_near_ood_benchmarks,
)
from naltra.data.preprocessing import (  # noqa: E402
    generate_multifin_leakage_free_track,
    process_mn_ds,
    process_multifin,
    process_sib200,
)


def prepare_sib200() -> None:
    print("\n" + "=" * 50)
    print("Preparing SIB-200 (English & Turkish)")
    print("=" * 50)
    splits = process_sib200()

    total_records = 0
    label_counter: Counter[str] = Counter()
    lang_counter: Counter[str] = Counter()

    print("\n=== PREPARATION SUMMARY (SIB-200) ===")
    for split_name, records in splits.items():
        total_records += len(records)
        print(f"Split '{split_name}': {len(records)} records")
        for r in records:
            lang_counter[r["language"]] += 1
            for lbl in r["labels"]:
                label_counter[lbl] += 1

    print(f"\nTotal processed records: {total_records}")
    print("\nLanguage breakdown:")
    for lang, count in lang_counter.items():
        print(f"  {lang}: {count}")

    print("\nCanonical label distribution:")
    for label, count in label_counter.most_common():
        print(f"  {label}: {count}")

    print("\nOutput location: data/processed/sib200/")
    print("All records validated successfully against taxonomy 0.2.0.\n")


def prepare_multifin() -> None:
    print("\n" + "=" * 50)
    print("Preparing MultiFin (English & Turkish)")
    print("=" * 50)
    splits = process_multifin()

    total_records = 0
    label_counter: Counter[str] = Counter()
    lang_counter: Counter[str] = Counter()
    multi_label_count = 0

    print("\n=== PREPARATION SUMMARY (MultiFin) ===")
    for split_name, records in splits.items():
        total_records += len(records)
        print(f"Split '{split_name}': {len(records)} records")
        for r in records:
            lang_counter[r["language"]] += 1
            if len(r["labels"]) > 1:
                multi_label_count += 1
            for lbl in r["labels"]:
                label_counter[lbl] += 1

    print(f"\nTotal processed records: {total_records}")
    print("\nLanguage breakdown:")
    for lang, count in lang_counter.items():
        print(f"  {lang}: {count}")

    pct = multi_label_count / total_records * 100
    print(f"\nMulti-label examples (>1 label): {multi_label_count} ({pct:.2f}%)")

    print("\nCanonical label distribution:")
    for label, count in label_counter.most_common():
        print(f"  {label}: {count}")

    print("\nOutput location: data/processed/multifin/")
    print("All records validated successfully against taxonomy 0.2.0.\n")

    print("\nGenerating MultiFin Leakage-Free Evaluation Track...")
    lf_stats = generate_multifin_leakage_free_track()
    print("=== MULTIFIN LEAKAGE-FREE SUMMARY ===")
    print(f"Train (official reference): {lf_stats['train_count']}")
    val_clean = lf_stats["clean_val_count"]
    val_rem = lf_stats["val_removed_count"]
    print(f"Validation (leakage-free): {val_clean} ({val_rem} leaks removed)")
    test_clean = lf_stats["clean_test_count"]
    test_rem = lf_stats["test_removed_count"]
    print(f"Test (leakage-free): {test_clean} ({test_rem} leaks removed)")
    print(f"Validation/Test Content Overlap: {lf_stats['val_test_overlap_count']}")
    print("Output location: data/splits/multifin/leakage_free/\n")


def prepare_mn_ds() -> None:
    print("\n" + "=" * 50)
    print("Preparing MN-DS (Hierarchical News Track)")
    print("=" * 50)
    splits = process_mn_ds()

    total_records = 0
    multi_label_count = 0
    split_labels: dict[str, set[str]] = {}

    print("\n=== PREPARATION SUMMARY (MN-DS) ===")
    for split_name, records in splits.items():
        total_records += len(records)
        split_labels[split_name] = {lbl for r in records for lbl in r["labels"]}
        num_lbls = len(split_labels[split_name])
        print(f"Split '{split_name}': {len(records)} records (unique labels: {num_lbls})")
        for r in records:
            if len(r["labels"]) > 1:
                multi_label_count += 1

    print(f"\nTotal unique articles: {total_records}")
    pct = multi_label_count / total_records * 100
    print(f"Multi-label articles (>1 label): {multi_label_count} ({pct:.2f}%)")

    all_labels = set().union(*split_labels.values())
    print(f"Total fine-grained labels covered across splits: {len(all_labels)}")
    print(f"Labels in train: {len(split_labels['train'])}/109")
    print(f"Labels in validation: {len(split_labels['validation'])}/109")
    print(f"Labels in test: {len(split_labels['test'])}/109")

    print("\nOutput location: data/processed/mn_ds/")
    print("All records validated successfully against taxonomy 0.2.0.\n")


def prepare_noisy() -> None:
    print("\n" + "=" * 50)
    print("Preparing Noisy Robustness Benchmark (Validation & Test)")
    print("=" * 50)
    strategy = "combined"
    severity = "medium"
    results = generate_noisy_benchmarks(strategy=strategy, severity=severity)

    total_noisy = sum(sum(v.values()) for v in results.values())
    print("\n=== NOISY BENCHMARK SUMMARY ===")
    print(f"Strategy: {strategy} (typos, capitalization, punctuation)")
    print(f"Severity: {severity} (10% perturbation probability)")
    print(f"Total noisy records generated: {total_noisy:,}")
    for ds_name, splits in results.items():
        print(f"  - {ds_name}: {splits}")

    sample_file = Path("data/processed/sib200/test.jsonl")
    if sample_file.exists():
        clean_recs = load_jsonl(sample_file)
        if clean_recs:
            orig = clean_recs[0]
            noisy = create_noisy_record(orig, strategy=strategy, severity=severity)
            print("\n--- Example Clean vs. Noisy Pair ---")
            print(f"ID:          {noisy['id']}")
            print(f"Original ID: {noisy['original_id']}")
            print(f"Language:    {noisy['language']} (unchanged)")
            print(f"Labels:      {noisy['labels']} (unchanged)")
            print(f"Clean Text:  {orig['text']}")
            print(f"Noisy Text:  {noisy['text']}")

    print(f"\nOutput location: data/noisy/{strategy}/{severity}/")
    print("Clean training sets remain untouched.\n")


def prepare_code_switch() -> None:
    print("\n" + "=" * 50)
    print("Preparing Synthetic EN/TR Code-Switch Benchmark")
    print("=" * 50)
    strategy = "chunk_mix"
    strength = "balanced"
    results = generate_code_switch_benchmarks(strategy=strategy, strength=strength)

    total_cs = sum(results.values())
    print("\n=== CODE-SWITCH BENCHMARK SUMMARY ===")
    print(f"Strategy: {strategy} (controlled synthetic chunk-mixing)")
    print(f"Strength: {strength} (~50/50 token mix from aligned pairs)")
    print(f"Total code-switched records: {total_cs:,}")
    for split_name, count in results.items():
        print(f"  - {split_name}: {count}")

    val_file = Path("data/processed/sib200/validation.jsonl")
    if val_file.exists():
        records = load_jsonl(val_file)
        pairs = pair_aligned_records(records)
        if pairs:
            first_pair_id = sorted(pairs.keys())[0]
            en_r, tr_r = pairs[first_pair_id]
            sample_cs = create_code_switched_record(
                en_r, tr_r, strategy=strategy, strength=strength
            )
            print("\n--- Example EN Source / TR Source / Mixed Record ---")
            print(f"Generated ID:    {sample_cs['id']}")
            print(f"Pair ID:         {sample_cs['pair_id']} (preserved)")
            print(f"EN Source ID:    {sample_cs['en_id']}")
            print(f"TR Source ID:    {sample_cs['tr_id']}")
            print(f"Language:        {sample_cs['language']}")
            print(f"Labels:          {sample_cs['labels']} (preserved)")
            print(f"EN Source Text:  {en_r['text']}")
            print(f"TR Source Text:  {tr_r['text']}")
            print(f"Synthetic Mixed: {sample_cs['text']}")

    print(f"\nOutput location: data/processed/code_switch/{strategy}/{strength}/")
    print("Clean training data remains untouched.\n")


def prepare_ood() -> None:
    print("\n" + "=" * 50)
    print("Preparing NALTRA Out-of-Distribution (OOD) Benchmarks")
    print("=" * 50)

    print("\n--- 1. Near-OOD: SIB-200 Leave-One-Topic-Out (7 Folds) ---")
    near_stats = generate_near_ood_benchmarks()
    print("Folds generated successfully:")
    for topic, stats in sorted(near_stats.items()):
        print(
            f"  - {topic}: train_id={stats['train_id']}, "
            f"val_ood={stats['validation_ood']}, test_ood={stats['test_ood']} "
            f"(total OOD={stats['total_ood']})"
        )

    print("\n--- 2. Far-OOD: Amazon MASSIVE (English & Turkish) ---")
    far_stats = generate_far_ood_benchmarks()
    print("Far-OOD generated successfully:")
    print(f"  - validation: {far_stats['validation']} records (250 EN, 250 TR)")
    print(f"  - test: {far_stats['test']} records (500 EN, 500 TR)")

    sample_near = Path("data/ood/near/sib200/politics/validation_ood.jsonl")
    if sample_near.exists():
        records = load_jsonl(sample_near)
        if records:
            r = records[0]
            print("\n--- Example Near-OOD Record ---")
            print(f"ID:           {r['id']}")
            print(f"OOD Type:     {r['ood_type']}")
            print(f"Held-out:     {r.get('source_label')}")
            print(f"Reason:       {r['ood_reason']}")
            print(f"Language:     {r['language']}")
            print(f"Text:         {r['text']}")

    sample_far = Path("data/ood/far/massive/test_ood.jsonl")
    if sample_far.exists():
        records = load_jsonl(sample_far)
        if len(records) >= 2:
            en_r = next((r for r in records if r["language"] == "en"), None)
            if en_r:
                tr_r = next(
                    (
                        r
                        for r in records
                        if r["language"] == "tr" and r["pair_id"] == en_r["pair_id"]
                    ),
                    None,
                )
                if tr_r:
                    print("\n--- Example Far-OOD EN/TR Matched Pair ---")
                    print(f"Pair ID:      {en_r['pair_id']}")
                    print(f"EN ID:        {en_r['id']}")
                    print(f"TR ID:        {tr_r['id']}")
                    scen = f"{en_r.get('scenario')}:{en_r.get('intent')}"
                    print(f"Scenario:     {scen}")
                    print(f"EN Text:      {en_r['text']}")
                    print(f"TR Text:      {tr_r['text']}")

    print("\nNear-OOD location: data/ood/near/sib200/")
    print("Far-OOD location:  data/ood/far/massive/")
    print("Zero training leakage. Clean training sets remain untouched.\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and preprocess datasets for NALTRA benchmark tracks."
    )
    parser.add_argument(
        "--dataset",
        choices=["sib200", "multifin", "mn_ds", "noisy", "code_switch", "ood", "all"],
        default="all",
        help="Dataset pipeline to run (default: all)",
    )
    args = parser.parse_args()

    if args.dataset in ("sib200", "all"):
        prepare_sib200()
    if args.dataset in ("multifin", "all"):
        prepare_multifin()
    if args.dataset in ("mn_ds", "all"):
        prepare_mn_ds()
    if args.dataset in ("noisy", "all"):
        prepare_noisy()
    if args.dataset in ("code_switch", "all"):
        prepare_code_switch()
    if args.dataset in ("ood", "all"):
        prepare_ood()


if __name__ == "__main__":
    main()
