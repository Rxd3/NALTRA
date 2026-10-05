from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from naltra.data.code_switching import generate_code_switch_benchmarks  # noqa: E402
from naltra.data.noise import generate_noisy_benchmarks  # noqa: E402
from naltra.data.ood import generate_near_ood_benchmarks  # noqa: E402
from naltra.data.preprocessing import process_sib200  # noqa: E402


def prepare_sib200() -> None:
    print("\n" + "=" * 50)
    print("Preparing SIB-200 (English & Turkish)")
    print("=" * 50)

    splits = process_sib200()

    total_records = 0
    label_counter: Counter[str] = Counter()
    lang_counter: Counter[str] = Counter()

    for split_name, records in splits.items():
        total_records += len(records)
        print(f"Split '{split_name}': {len(records)} records")

        for record in records:
            lang_counter[record["language"]] += 1
            for label in record["labels"]:
                label_counter[label] += 1

    print(f"\nTotal records: {total_records}")
    print(f"Languages: {dict(lang_counter)}")
    print(f"Labels: {dict(label_counter)}")
    print("\nOutput: data/processed/sib200/")


def prepare_noisy() -> None:
    print("\n" + "=" * 50)
    print("Preparing SIB-200 Noisy Robustness Benchmark")
    print("=" * 50)

    strategy = "combined"
    severity = "medium"

    results = generate_noisy_benchmarks(
        datasets=["sib200"],
        strategy=strategy,
        severity=severity,
    )

    print(f"\nGenerated noisy SIB-200 benchmark: {results}")
    print(f"Output: data/noisy/{strategy}/{severity}/sib200/")


def prepare_code_switch() -> None:
    print("\n" + "=" * 50)
    print("Preparing SIB-200 EN/TR Code-Switch Benchmark")
    print("=" * 50)

    strategy = "chunk_mix"
    strength = "balanced"

    results = generate_code_switch_benchmarks(
        strategy=strategy,
        strength=strength,
    )

    print(f"\nGenerated code-switched benchmark: {results}")
    print(f"Output: data/processed/code_switch/{strategy}/{strength}/")


def prepare_ood() -> None:
    print("\n" + "=" * 50)
    print("Preparing SIB-200 Near-OOD Benchmark")
    print("=" * 50)

    results = generate_near_ood_benchmarks()

    print("\nGenerated SIB-200 leave-one-topic-out folds:")
    for topic, stats in sorted(results.items()):
        print(
            f"  {topic}: "
            f"train={stats['train_id']}, "
            f"validation_ood={stats['validation_ood']}, "
            f"test_ood={stats['test_ood']}"
        )

    print("\nOutput: data/ood/near/sib200/")


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Prepare SIB-200 and derived NALTRA benchmark tracks."
    )

    parser.add_argument(
        "--dataset",
        choices=["sib200", "noisy", "code_switch", "ood", "all"],
        default="all",
    )

    args = parser.parse_args()

    if args.dataset in ("sib200", "all"):
        prepare_sib200()

    if args.dataset in ("noisy", "all"):
        prepare_noisy()

    if args.dataset in ("code_switch", "all"):
        prepare_code_switch()

    if args.dataset in ("ood", "all"):
        prepare_ood()


if __name__ == "__main__":
    main()
