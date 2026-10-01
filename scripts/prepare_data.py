from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

# Ensure src/ is on sys.path so naltra can be imported directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NALTRA benchmark datasets.")
    parser.add_argument(
        "--dataset",
        choices=["sib200", "multifin", "mn_ds", "all"],
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


if __name__ == "__main__":
    main()
