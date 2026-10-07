"""Dataset loading, preparation, augmentation, and splitting utilities."""

from naltra.data.code_switching import (
    create_code_switched_record,
    generate_code_switch_benchmarks,
    mix_code_switched_text,
    pair_aligned_records,
)
from naltra.data.loader import load_jsonl, save_jsonl
from naltra.data.manifest import (
    compute_file_md5,
    compute_file_sha256,
    create_manifest,
)
from naltra.data.noise import create_noisy_record, generate_noisy_benchmarks, perturb_text
from naltra.data.ood import (
    generate_near_ood_benchmarks,
    validate_ood_record,
)
from naltra.data.preprocessing import (
    compute_content_fingerprint,
    load_canonical_label_ids,
    load_label_map,
    load_taxonomy_parents,
    map_labels,
    preprocess_record_text,
    process_cordis_h2020,
    validate_record,
)
from naltra.data.splits import (
    grouped_multilabel_stratified_split,
    multilabel_stratified_split,
    split_records,
)

__all__ = [
    "compute_content_fingerprint",
    "compute_file_md5",
    "compute_file_sha256",
    "create_code_switched_record",
    "create_manifest",
    "create_noisy_record",
    "generate_code_switch_benchmarks",
    "generate_near_ood_benchmarks",
    "generate_noisy_benchmarks",
    "grouped_multilabel_stratified_split",
    "load_canonical_label_ids",
    "load_jsonl",
    "load_label_map",
    "load_taxonomy_parents",
    "map_labels",
    "mix_code_switched_text",
    "multilabel_stratified_split",
    "pair_aligned_records",
    "perturb_text",
    "preprocess_record_text",
    "process_cordis_h2020",
    "save_jsonl",
    "split_records",
    "validate_ood_record",
    "validate_record",
]
