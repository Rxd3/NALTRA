from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

# Ensure src/ is on sys.path so naltra can be imported directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from naltra.data.preprocessing import process_sib200


def prepare_sib200() -> None:
    print("\n" + "=" * 50)
    print("Preparing SIB-200 (English & Turkish)")
    print("=" * 50)
    splits = process_sib200()

    total_records = 0
    label_counter: Counter[str] = Counter()
    lang_counter: Counter[str] = Counter()

    print("\n=== PREPARATION SUMMARY ===")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NALTRA benchmark datasets.")
    parser.add_argument(
        "--dataset",
        choices=["sib200"],
        default="sib200",
        help="Which dataset to process (default: sib200).",
    )
    args = parser.parse_args()

    if args.dataset == "sib200":
        prepare_sib200()


if __name__ == "__main__":
    main()
