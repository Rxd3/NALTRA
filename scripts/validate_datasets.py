"""Audit CORDIS provenance, contamination, and derived-source fidelity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from naltra.data.manifest import GenerationCodeMismatch, validate_manifest  # noqa: E402
from naltra.data.noise import create_noisy_record  # noqa: E402
from naltra.data.validation import (  # noqa: E402, F401
    SPLITS,
    check_disjoint_partitions,
    check_input_inventory,
    load_audit_records,
    validate_code_switch_release,
    validate_cordis_release,
    validate_prebuilt_cordis_release,
)

RAW_SOURCES = ("project.csv", "euroSciVoc.csv")
PREBUILT_REMEDY = "audit the shipped release records with --prebuilt-release."


def validate_derived(release: dict) -> None:
    base = REPO_ROOT / "data/processed/cordis_h2020"
    taxonomy = REPO_ROOT / "taxonomy/taxonomy.json"
    # The layout prepare_data.py --dataset noisy writes: <noisy>/cordis_h2020/{en,tr}/test.jsonl.
    noisy = REPO_ROOT / "data/noisy/combined/medium"
    languages = ("en", "tr")
    for directory, name, filenames, sources in (
        *(
            (noisy / "cordis_h2020" / lang, f"noisy_cordis_h2020/{lang}", ["test.jsonl"], [lang])
            for lang in languages
        ),
        (
            noisy,
            "noisy_robustness_benchmark",
            [f"cordis_h2020/{lang}/test.jsonl" for lang in languages],
            languages,
        ),
    ):
        manifest = validate_manifest(directory, name, filenames, taxonomy.parent)
        check_input_inventory(directory, manifest, [base / lang / "test.jsonl" for lang in sources])
        parameters = manifest["generation_parameters"]
        if any(
            parameters.get(k) != v
            for k, v in {
                "strategy": "combined",
                "severity": "medium",
                "base_seed": 42,
                "splits": ["test"],
            }.items()
        ):
            raise ValueError("Noise parameters differ from the benchmark specification.")
    for language in languages:
        expected = [
            create_noisy_record(r, strategy="combined", severity="medium", base_seed=42)
            for r in release["corpora"][language]["test"]
        ]
        path = noisy / "cordis_h2020" / language / "test.jsonl"
        if load_audit_records(path, taxonomy) != expected:
            raise ValueError(f"Noisy {language} test changed clean-source fidelity.")
    validate_code_switch_release(base, release, taxonomy_path=taxonomy)


def describe_failure(exc: Exception) -> str:
    """Name the remedy when the full audit cannot lock raw sources or generation code."""
    name = Path(str(getattr(exc, "filename", ""))).name
    if isinstance(exc, FileNotFoundError) and name in RAW_SOURCES:
        return (
            f"raw CORDIS source {name} is missing. Run `python scripts/prepare_data.py "
            f"--dataset cordis_h2020 --download` first, or {PREBUILT_REMEDY}"
        )
    if isinstance(exc, GenerationCodeMismatch):
        return (
            f"{exc} The data-generation code changed since this release was built. "
            f"Regenerate it with `python scripts/prepare_data.py`, or {PREBUILT_REMEDY}"
        )
    return str(exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", choices=("en", "tr"), default=["en", "tr"])
    parser.add_argument(
        "--code-switch",
        action="store_true",
        help="Audit both code-switch tracks without requiring the optional noise dataset.",
    )
    parser.add_argument(
        "--derived",
        action="store_true",
        help="Also require/audit noise and both code-switch tracks.",
    )
    parser.add_argument(
        "--prebuilt-release",
        action="store_true",
        help="Audit the shipped prebuilt EN/TR release records as train_all.py does; raw "
        "sources and generation-code hashes are not required. Not combinable with "
        "--code-switch or --derived.",
    )
    args = parser.parse_args(argv)
    if (args.derived or args.code_switch) and set(args.languages) != {"en", "tr"}:
        parser.error("Derived/code-switch audits require both en and tr.")
    if args.prebuilt_release and (args.derived or args.code_switch):
        parser.error(
            "--prebuilt-release audits the EN and TR records only; the noise and code-switch "
            "audits lock generation-code hashes and need the full audit."
        )
    try:
        audit = (
            validate_prebuilt_cordis_release if args.prebuilt_release else validate_cordis_release
        )
        release = audit(
            REPO_ROOT / "data/processed/cordis_h2020",
            args.languages,
            taxonomy_path=REPO_ROOT / "taxonomy/taxonomy.json",
        )
        verified = "prebuilt records" if args.prebuilt_release else "release integrity"
        for language in args.languages:
            counts = {s: len(release["corpora"][language][s]) for s in SPLITS}
            print(f"[PASS] CORDIS {language}: {counts}; {verified} verified.")
        summary = "AUDIT SUMMARY: PASSED"
        if issues := release.get("translation_issues"):
            print(
                "[INFO] Translation issues reported, not enforced: segmenter drift "
                f"{issues['segmenter_drift']}, alignment/text mismatches "
                f"{len(issues['alignment_text_mismatch'])}, QA failures "
                f"{len(issues['qa_failed'])}."
            )
            summary += (
                " (prebuilt records; alignment/text mismatches and QA failures are reported, "
                "not enforced)"
            )
        if args.derived:
            validate_derived(release)
            print("[PASS] CORDIS noise and code-switch source fidelity.")
        elif args.code_switch:
            mixed = validate_code_switch_release(REPO_ROOT / "data/processed/cordis_h2020", release)
            for strategy, partitions in mixed["corpora"].items():
                counts = {split: len(records) for split, records in partitions.items()}
                print(f"[PASS] Code-switch {strategy}: {counts}; source fidelity verified.")
        print(summary)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[FAIL] CORDIS release: {describe_failure(exc)}")
        print("AUDIT SUMMARY: FAILED. Prepare/regenerate the declared release before training.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
