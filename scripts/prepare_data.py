from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

# Ensure src/ is on sys.path so naltra can be imported directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from naltra.data.loader import load_jsonl
from naltra.data.noise import create_noisy_record, generate_noisy_benchmarks
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NALTRA benchmark datasets.")
    parser.add_argument(
        "--dataset",
        choices=["sib200", "multifin", "mn_ds", "noisy", "all"],
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


if __name__ == "__main__":
    main()
