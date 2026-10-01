"""Dataset loading, preparation, augmentation, and splitting utilities."""

from naltra.data.code_switching import (
    create_code_switched_record,
    generate_code_switch_benchmarks,
    mix_code_switched_text,
    pair_aligned_records,
)
from naltra.data.loader import load_jsonl, save_jsonl
from naltra.data.noise import create_noisy_record, generate_noisy_benchmarks, perturb_text
from naltra.data.ood import (
    MASSIVE_ALLOWED_SCENARIOS,
    generate_far_ood_benchmarks,
    generate_near_ood_benchmarks,
    validate_ood_record,
)
from naltra.data.preprocessing import (
    load_canonical_label_ids,
    load_label_map,
    map_labels,
    preprocess_record_text,
    process_mn_ds,
    process_multifin,
    process_sib200,
    validate_record,
)
from naltra.data.splits import multilabel_stratified_split, split_records

__all__ = [
    "MASSIVE_ALLOWED_SCENARIOS",
    "create_code_switched_record",
    "create_noisy_record",
    "generate_code_switch_benchmarks",
    "generate_far_ood_benchmarks",
    "generate_near_ood_benchmarks",
    "generate_noisy_benchmarks",
    "load_canonical_label_ids",
    "load_jsonl",
    "load_label_map",
    "map_labels",
    "mix_code_switched_text",
    "multilabel_stratified_split",
    "pair_aligned_records",
    "perturb_text",
    "preprocess_record_text",
    "process_mn_ds",
    "process_multifin",
    "process_sib200",
    "save_jsonl",
    "split_records",
    "validate_ood_record",
    "validate_record",
]
