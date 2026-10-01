from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

# Ensure src/ is on sys.path so naltra can be imported directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from naltra.data.code_switching import (
    create_code_switched_record,
    generate_code_switch_benchmarks,
    pair_aligned_records,
)
from naltra.data.loader import load_jsonl
from naltra.data.noise import create_noisy_record, generate_noisy_benchmarks
from naltra.data.ood import (
    MASSIVE_ALLOWED_SCENARIOS,
    generate_far_ood_benchmarks,
    generate_near_ood_benchmarks,
)
from naltra.data.preprocessing import process_mn_ds, process_multifin, process_sib200


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
            for l in r["labels"]:
                label_counter[l] += 1

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
            for l in r["labels"]:
                label_counter[l] += 1

    print(f"\nTotal processed records: {total_records}")
    print("\nLanguage breakdown:")
    for lang, count in lang_counter.items():
        print(f"  {lang}: {count}")

    print(f"\nMulti-label examples (>1 label): {multi_label_count} ({multi_label_count / total_records * 100:.2f}%)")

    print("\nCanonical label distribution:")
    for label, count in label_counter.most_common():
        print(f"  {label}: {count}")

    print("\nOutput location: data/processed/multifin/")
    print("All records validated successfully against taxonomy 0.2.0.\n")


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
        split_labels[split_name] = {l for r in records for l in r["labels"]}
        print(f"Split '{split_name}': {len(records)} records (unique labels: {len(split_labels[split_name])})")
        for r in records:
            if len(r["labels"]) > 1:
                multi_label_count += 1

    print(f"\nTotal unique articles: {total_records}")
    print(f"Multi-label articles (>1 label): {multi_label_count} ({multi_label_count / total_records * 100:.2f}%)")

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

    # Display exactly one clean/noisy pair example
    sample_file = Path("data/processed/sib200/test.jsonl")
    if sample_file.exists():
        clean_sample = load_jsonl(sample_file)[0]
        noisy_sample = create_noisy_record(clean_sample, strategy=strategy, severity=severity)
        print("\n--- Example Clean vs. Noisy Pair ---")
        print(f"ID:          {noisy_sample['id']}")
        print(f"Original ID: {noisy_sample['original_id']}")
        print(f"Language:    {noisy_sample['language']} (unchanged)")
        print(f"Labels:      {noisy_sample['labels']} (unchanged)")
        print(f"Clean Text:  {clean_sample['text']}")
        print(f"Noisy Text:  {noisy_sample['text']}")

    print(f"\nOutput location: data/noisy/{strategy}/{severity}/")
    print("Clean training sets remain untouched.\n")


def prepare_code_switch() -> None:
    print("\n" + "=" * 50)
    print("Preparing Synthetic EN/TR Code-Switch Benchmark")
    print("=" * 50)
    strategy = "chunk_mix"
    strength = "balanced"
    results = generate_code_switch_benchmarks(strategy=strategy, strength=strength)

    total_records = sum(results.values())
    print("\n=== CODE-SWITCH BENCHMARK SUMMARY ===")
    print(f"Strategy: {strategy} (controlled synthetic chunk-mixing)")
    print(f"Strength: {strength} (~50/50 token mix from aligned pairs)")
    print(f"Total code-switched records: {total_records}")
    print(f"  - validation: {results.get('validation', 0)}")
    print(f"  - test: {results.get('test', 0)}")

    # Display exactly one EN source / TR source / mixed example
    val_file = Path("data/processed/sib200/validation.jsonl")
    if val_file.exists():
        pairs = pair_aligned_records(load_jsonl(val_file))
        if pairs:
            en_sample, tr_sample = pairs[0]
            mixed_sample = create_code_switched_record(
                en_sample, tr_sample, strategy=strategy, strength=strength
            )
            print("\n--- Example EN Source / TR Source / Mixed Record ---")
            print(f"Generated ID:    {mixed_sample['id']}")
            print(f"Pair ID:         {mixed_sample['pair_id']} (preserved)")
            print(f"EN Source ID:    {mixed_sample['en_id']}")
            print(f"TR Source ID:    {mixed_sample['tr_id']}")
            print(f"Language:        {mixed_sample['language']}")
            print(f"Labels:          {mixed_sample['labels']} (preserved)")
            print(f"EN Source Text:  {en_sample['text']}")
            print(f"TR Source Text:  {tr_sample['text']}")
            print(f"Synthetic Mixed: {mixed_sample['text']}")

    print(f"\nOutput location: data/processed/code_switch/{strategy}/{strength}/")
    print("Clean training data remains untouched.\n")


def prepare_ood() -> None:
    print("\n" + "=" * 50)
    print("Preparing NALTRA Out-of-Distribution (OOD) Benchmarks")
    print("=" * 50)

    # 1. Near-OOD: SIB-200 leave-one-topic-out (7 folds)
    print("\n--- 1. Near-OOD: SIB-200 Leave-One-Topic-Out (7 Folds) ---")
    near_stats = generate_near_ood_benchmarks()
    print("Folds generated successfully:")
    for topic, stats in near_stats.items():
        print(
            f"  - {topic}: train_id={stats['train_id']}, "
            f"val_ood={stats['validation_ood']}, test_ood={stats['test_ood']} "
            f"(total OOD={stats['total_ood']})"
        )

    # 2. Far-OOD: Amazon MASSIVE (EN & TR matched pairs)
    print("\n--- 2. Far-OOD: Amazon MASSIVE (English & Turkish) ---")
    far_stats = generate_far_ood_benchmarks()
    print("Far-OOD generated successfully:")
    for split_name, count in far_stats.items():
        print(f"  - {split_name}: {count} records ({count // 2} EN, {count // 2} TR)")

    # Display one Near-OOD example and one Far-OOD EN/TR pair example
    sample_near_file = Path("data/ood/near/sib200/politics/validation_ood.jsonl")
    if sample_near_file.exists():
        near_sample = load_jsonl(sample_near_file)[0]
        print("\n--- Example Near-OOD Record ---")
        print(f"ID:           {near_sample['id']}")
        print(f"OOD Type:     {near_sample['ood_type']}")
        print(f"Held-out:     {near_sample['source_label']}")
        print(f"Reason:       {near_sample['ood_reason']}")
        print(f"Language:     {near_sample['language']}")
        print(f"Text:         {near_sample['text']}")

    sample_far_file = Path("data/ood/far/massive/test_ood.jsonl")
    if sample_far_file.exists():
        far_recs = load_jsonl(sample_far_file)
        en_far = far_recs[0]
        tr_far = far_recs[1]
        print("\n--- Example Far-OOD EN/TR Matched Pair ---")
        print(f"Pair ID:      {en_far['pair_id']}")
        print(f"EN ID:        {en_far['id']}")
        print(f"TR ID:        {tr_far['id']}")
        print(f"Scenario:     {en_far['source_label']}")
        print(f"EN Text:      {en_far['text']}")
        print(f"TR Text:      {tr_far['text']}")

    print("\nNear-OOD location: data/ood/near/sib200/")
    print("Far-OOD location:  data/ood/far/massive/")
    print("Zero training leakage. Clean training sets remain untouched.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NALTRA benchmark datasets.")
    parser.add_argument(
        "--dataset",
        choices=["sib200", "multifin", "mn_ds", "noisy", "code_switch", "ood", "all"],
        default="all",
        help="Which dataset to process (default: all).",
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
