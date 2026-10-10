"""Installed and script entry points share the same CORDIS preparation workflow."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from naltra.data.manifest import REPO_ROOT


def prepare_data(argv: Sequence[str] | None = None) -> None:
    from naltra.data.code_switching import generate_code_switch_benchmarks
    from naltra.data.noise import generate_noisy_benchmarks
    from naltra.data.preprocessing import download_cordis_h2020, process_cordis_h2020
    from naltra.data.translation import HuggingFaceLocalTranslator, translate_cordis_dataset
    from naltra.data.validation import validate_cordis_release, validate_prebuilt_cordis_release

    parser = argparse.ArgumentParser(description="Prepare CORDIS H2020 and derived EN/TR tracks.")
    parser.add_argument(
        "--dataset",
        choices=("cordis_h2020", "translation", "noisy", "code_switch", "all"),
        default="cordis_h2020",
    )
    parser.add_argument(
        "--download", action="store_true", help="Fetch the locked official raw archive if absent."
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument(
        "--batch-size", type=int, default=24, help="Translation sentence batch size."
    )
    parser.add_argument(
        "--prebuilt-release",
        action="store_true",
        help="Audit the shipped prebuilt release records (as train_all.py) before deriving tracks.",
    )
    parser.add_argument(
        "--noise-dir",
        type=Path,
        default=REPO_ROOT / "data/noisy",
        help="Noisy output root, on the same drive as data/; files land in "
        "<dir>/combined/medium/cordis_h2020/{en,tr}.",
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    root = REPO_ROOT / "data"
    base = root / "processed/cordis_h2020"
    taxonomy = REPO_ROOT / "taxonomy/taxonomy.json"
    validate_derivation_source = (
        validate_prebuilt_cordis_release if args.prebuilt_release else validate_cordis_release
    )
    if args.dataset in ("cordis_h2020", "all"):
        raw = root / "raw/cordis_h2020"
        if args.download:
            download_cordis_h2020(raw)
        process_cordis_h2020(
            raw,
            base / "en",
            root / "splits/cordis_h2020",
            taxonomy,
            taxonomy.parent / "label_map.json",
        )
    if args.dataset in ("translation", "all"):
        validate_cordis_release(base, ["en"])
        translate_cordis_dataset(
            base / "en",
            base / "tr",
            root / "cache/translations/cordis_h2020",
            taxonomy,
            translator=HuggingFaceLocalTranslator(device=args.device, batch_size=args.batch_size),
            review_sample_path=root / "review/translation_review_sample.csv",
        )
    if args.dataset in ("noisy", "all"):
        validate_derivation_source(base, ["en", "tr"])
        generate_noisy_benchmarks(
            processed_base_dir=root / "processed", output_base_dir=args.noise_dir
        )
    if args.dataset in ("code_switch", "all"):
        validate_derivation_source(base, ["en", "tr"])
        for strategy in ("sentence_mix", "chunk_mix"):
            generate_code_switch_benchmarks(
                en_dir=base / "en",
                tr_dir=base / "tr",
                output_base_dir=base / "code_switch",
                strategy=strategy,
                taxonomy_path=taxonomy,
            )
