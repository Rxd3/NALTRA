"""Small console entry points for the installed package."""

import argparse


def prepare_data() -> None:
    """Run the same dataset generators from the installed console command."""
    from naltra.data.code_switching import generate_code_switch_benchmarks
    from naltra.data.noise import generate_noisy_benchmarks
    from naltra.data.ood import generate_far_ood_benchmarks, generate_near_ood_benchmarks
    from naltra.data.preprocessing import (
        generate_multifin_leakage_free_track,
        process_mn_ds,
        process_multifin,
        process_sib200,
    )

    parser = argparse.ArgumentParser(description="Prepare NALTRA benchmark datasets.")
    parser.add_argument(
        "--dataset",
        choices=("sib200", "multifin", "mn_ds", "noisy", "code_switch", "ood", "all"),
        default="all",
    )
    args = parser.parse_args()
    steps = {
        "sib200": (process_sib200,),
        "multifin": (process_multifin, generate_multifin_leakage_free_track),
        "mn_ds": (process_mn_ds,),
        "noisy": (generate_noisy_benchmarks,),
        "code_switch": (generate_code_switch_benchmarks,),
        "ood": (generate_near_ood_benchmarks, generate_far_ood_benchmarks),
    }
    for dataset, generators in steps.items():
        if args.dataset in (dataset, "all"):
            for generate in generators:
                generate()
