"""Audit CORDIS provenance, contamination, and derived-source fidelity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from naltra.data.code_switching import create_code_switched_record  # noqa: E402
from naltra.data.manifest import validate_manifest  # noqa: E402
from naltra.data.noise import create_noisy_record  # noqa: E402
from naltra.data.validation import (  # noqa: E402, F401
    SPLITS,
    check_disjoint_partitions,
    check_input_inventory,
    load_audit_records,
    validate_cordis_release,
)


def validate_derived(release: dict) -> None:
    base = REPO_ROOT / "data/processed/cordis_h2020"
    taxonomy = REPO_ROOT / "taxonomy/taxonomy.json"
    noisy = REPO_ROOT / "data/noisy/cordis_h2020/combined/medium"
    for directory, name, filenames in (
        (noisy / "en", "noisy_en", ["validation.jsonl", "test.jsonl"]),
        (noisy, "noisy_robustness_benchmark", ["en/validation.jsonl", "en/test.jsonl"]),
    ):
        manifest = validate_manifest(directory, name, filenames, taxonomy.parent)
        check_input_inventory(
            directory, manifest, [base / "en" / f"{s}.jsonl" for s in ("validation", "test")]
        )
        parameters = manifest["generation_parameters"]
        if any(
            parameters.get(k) != v
            for k, v in {
                "strategy": "combined",
                "severity": "medium",
                "base_seed": 42,
                "splits": ["validation", "test"],
            }.items()
        ):
            raise ValueError("Noise parameters differ from the benchmark specification.")
    for split in ("validation", "test"):
        expected = [
            create_noisy_record(r, strategy="combined", severity="medium", base_seed=42)
            for r in release["corpora"]["en"][split]
        ]
        if load_audit_records(noisy / "en" / f"{split}.jsonl", taxonomy) != expected:
            raise ValueError(f"Noisy {split} changed clean-source fidelity.")
    for strategy in ("sentence_mix", "chunk_mix"):
        directory = base / "code_switch" / strategy / "balanced"
        manifest = validate_manifest(
            directory,
            "cordis_h2020_code_switch",
            ["validation.jsonl", "test.jsonl"],
            taxonomy.parent,
        )
        check_input_inventory(
            directory,
            manifest,
            [
                base / language / f"{s}.jsonl"
                for language in ("en", "tr")
                for s in ("validation", "test")
            ],
        )
        parameters = manifest["generation_parameters"]
        if parameters != {
            "strategy": strategy,
            "strength": "balanced",
            "base_seed": 42,
            "splits": ["validation", "test"],
        }:
            raise ValueError("Code-switch parameters differ from the benchmark specification.")
        for split in ("validation", "test"):
            english = {r["pair_id"]: r for r in release["corpora"]["en"][split]}
            turkish = {r["pair_id"]: r for r in release["corpora"]["tr"][split]}
            expected = [
                create_code_switched_record(english[p], turkish[p], strategy=strategy)
                for p in sorted(english)
            ]
            if load_audit_records(directory / f"{split}.jsonl", taxonomy) != expected:
                raise ValueError(f"Code-switch {strategy}/{split} changed source fidelity.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", choices=("en", "tr"), default=["en", "tr"])
    parser.add_argument(
        "--derived",
        action="store_true",
        help="Also require/audit noise and both code-switch tracks.",
    )
    args = parser.parse_args(argv)
    if args.derived and set(args.languages) != {"en", "tr"}:
        parser.error("--derived requires both en and tr.")
    try:
        release = validate_cordis_release(
            REPO_ROOT / "data/processed/cordis_h2020",
            args.languages,
            taxonomy_path=REPO_ROOT / "taxonomy/taxonomy.json",
        )
        for language in args.languages:
            counts = {s: len(release["corpora"][language][s]) for s in SPLITS}
            print(f"[PASS] CORDIS {language}: {counts}; release integrity verified.")
        if args.derived:
            validate_derived(release)
            print("[PASS] CORDIS noise and code-switch source fidelity.")
        print("AUDIT SUMMARY: PASSED")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[FAIL] CORDIS release: {exc}")
        print("AUDIT SUMMARY: FAILED. Prepare/regenerate the declared release before training.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
