"""Deterministic synthetic EN/TR code-switch benchmark generation for CORDIS H2020.

Derives controlled multilingual code-switched evaluation variants from aligned English
and Turkish versions of the exact same CORDIS project ID. Code-switched records strictly
preserve project ID, split assignment, direct labels, and hierarchy-closed target sets.
"""

from __future__ import annotations

import copy
import hashlib
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from naltra.data.loader import load_jsonl, save_jsonl
from naltra.data.manifest import create_manifest, relative_path

VALID_CODE_SWITCH_STRATEGIES: set[str] = {
    "sentence_mix",
    "chunk_mix",
    "interleave",
}

VALID_CODE_SWITCH_STRENGTHS: set[str] = {
    "balanced",
    "light",
}


def get_code_switch_seed(
    pair_id: str,
    strategy: str,
    strength: str,
    base_seed: int = 42,
) -> int:
    """Generate a platform-independent integer seed via SHA-256."""
    key = f"{pair_id}:{strategy}:{strength}:{base_seed}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big")


def pair_aligned_records(
    en_records: Sequence[dict[str, Any]],
    tr_records: Sequence[dict[str, Any]] | None = None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Pair English and Turkish records by pair_id with strict verification."""
    if tr_records is None:
        all_recs = en_records
        en_list = [r for r in all_recs if r.get("language") == "en"]
        tr_list = [r for r in all_recs if r.get("language") == "tr"]
        en_pairs = {r.get("pair_id") for r in en_list if r.get("pair_id")}
        tr_pairs = {r.get("pair_id") for r in tr_list if r.get("pair_id")}
        all_pair_ids = en_pairs.union(tr_pairs)
        if len(en_list) != len(tr_list) or len(en_pairs.symmetric_difference(tr_pairs)) > 0:
            raise ValueError(
                f"Incomplete pair: found {len(en_list)} EN records and {len(tr_list)} TR records "
                f"across {len(all_pair_ids)} pairs."
            )
        return pair_aligned_records(en_list, tr_list)

    en_by_pair: dict[str, dict[str, Any]] = {}
    for r in en_records:
        pid = r.get("pair_id")
        if not pid:
            raise ValueError(f"Record {r.get('id')} is missing 'pair_id'.")
        if pid in en_by_pair:
            raise ValueError(f"Duplicate EN record for pair_id '{pid}': {r.get('id')}")
        en_by_pair[pid] = r

    tr_by_pair: dict[str, dict[str, Any]] = {}
    for r in tr_records:
        pid = r.get("pair_id")
        if not pid:
            raise ValueError(f"Record {r.get('id')} is missing 'pair_id'.")
        if pid in tr_by_pair:
            raise ValueError(f"Duplicate TR record for pair_id '{pid}': {r.get('id')}")
        tr_by_pair[pid] = r

    if set(en_by_pair) != set(tr_by_pair):
        raise ValueError("Incomplete pair: English and Turkish pair inventories differ.")
    common_pairs = sorted(en_by_pair)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for pid in common_pairs:
        en_rec = en_by_pair[pid]
        tr_rec = tr_by_pair[pid]

        if en_rec.get("split") != tr_rec.get("split"):
            raise ValueError(
                f"Split mismatch for pair_id '{pid}': "
                f"en={en_rec.get('split')}, tr={tr_rec.get('split')}"
            )
        if en_rec.get("labels") != tr_rec.get("labels"):
            raise ValueError(
                f"Label mismatch for pair_id '{pid}': "
                f"en={en_rec.get('labels')}, tr={tr_rec.get('labels')}"
            )
        if en_rec.get("labels_direct") != tr_rec.get("labels_direct"):
            raise ValueError(f"Direct label mismatch for pair_id '{pid}'.")
        pairs.append((en_rec, tr_rec))

    return pairs


def mix_aligned_sentences(
    sentence_pairs: list[dict[str, Any]],
    strategy: str = "sentence_mix",
    strength: str = "balanced",
    seed: int = 42,
    pair_id: str = "sample",
) -> tuple[str, str]:
    """Mix aligned sentence pairs into a coherent code-switched document."""
    if not sentence_pairs:
        raise ValueError("Code-switching requires non-empty aligned sentences.")
    if strategy not in VALID_CODE_SWITCH_STRATEGIES or strength not in VALID_CODE_SWITCH_STRENGTHS:
        raise ValueError("Unknown code-switch strategy or strength.")
    exact_seed = get_code_switch_seed(pair_id, strategy, strength, base_seed=seed)
    rng = random.Random(exact_seed)

    primary_lang = "en" if rng.random() < 0.5 else "tr"
    n_sentences = len(sentence_pairs)

    if n_sentences == 1:
        # Single-sentence fallback: chunk-mix tokens within the single sentence
        sp = sentence_pairs[0]
        en_toks = sp["en"].split()
        tr_toks = sp["tr"].split()
        if primary_lang == "en":
            cut_p = max(1, len(en_toks) // 2)
            cut_s = max(1, len(tr_toks) // 2)
            mixed_toks = en_toks[:cut_p] + tr_toks[cut_s:]
        else:
            cut_p = max(1, len(tr_toks) // 2)
            cut_s = max(1, len(en_toks) // 2)
            mixed_toks = tr_toks[:cut_p] + en_toks[cut_s:]
        return " ".join(mixed_toks).strip(), primary_lang

    mixed_sentences: list[str] = []
    if strategy in ("sentence_mix", "interleave"):
        # Interleave sentences based on strength
        for i, sp in enumerate(sentence_pairs):
            if strength == "balanced":
                use_primary = i % 2 == 0
            elif strength == "light":
                use_primary = i % 3 != 1
            else:
                use_primary = True

            chosen_lang = primary_lang if use_primary else ("tr" if primary_lang == "en" else "en")
            mixed_sentences.append(sp[chosen_lang])

    elif strategy == "chunk_mix":
        # First chunk from primary language, second chunk from secondary
        split_point = (
            max(1, n_sentences // 2) if strength == "balanced" else max(1, (3 * n_sentences) // 4)
        )
        sec_lang = "tr" if primary_lang == "en" else "en"
        for i, sp in enumerate(sentence_pairs):
            lang = primary_lang if i < split_point else sec_lang
            mixed_sentences.append(sp[lang])
    else:
        raise ValueError(f"Unknown code-switch strategy: '{strategy}'")

    return " ".join(mixed_sentences).strip(), primary_lang


def mix_code_switched_text(
    en_text: str,
    tr_text: str,
    strategy: str = "chunk_mix",
    strength: str = "balanced",
    seed: int = 42,
    pair_id: str = "sample",
) -> tuple[str, str]:
    """Mix two texts deterministically by sentences or chunks."""
    if strategy not in VALID_CODE_SWITCH_STRATEGIES:
        raise ValueError(f"Unknown code-switch strategy: '{strategy}'")
    if strength not in VALID_CODE_SWITCH_STRENGTHS:
        raise ValueError(f"Unknown code-switch strength: '{strength}'")
    if not en_text or not en_text.strip():
        raise ValueError("English text is empty or whitespace-only.")
    if not tr_text or not tr_text.strip():
        raise ValueError("Turkish text is empty or whitespace-only.")

    from naltra.data.translation import segment_sentences

    en_sents = segment_sentences(en_text)
    tr_sents = segment_sentences(tr_text)
    min_len = min(len(en_sents), len(tr_sents))
    sentence_pairs = [{"index": i, "en": en_sents[i], "tr": tr_sents[i]} for i in range(min_len)]
    if sentence_pairs:
        return mix_aligned_sentences(
            sentence_pairs=sentence_pairs,
            strategy=strategy,
            strength=strength,
            seed=seed,
            pair_id=pair_id,
        )

    # Fallback to token mixing if sentences are empty
    en_toks = en_text.split()
    tr_toks = tr_text.split()
    cut_en = max(1, len(en_toks) // 2)
    cut_tr = max(1, len(tr_toks) // 2)
    return " ".join(en_toks[:cut_en] + tr_toks[cut_tr:]).strip(), "en"


def create_code_switched_record(
    en_record: dict[str, Any],
    tr_record: dict[str, Any],
    strategy: str = "sentence_mix",
    strength: str = "balanced",
    base_seed: int = 42,
) -> dict[str, Any]:
    """Create a synthetic code-switched record from aligned EN and TR project variants."""
    pair_id = en_record.get("pair_id")
    if not pair_id or pair_id != tr_record.get("pair_id"):
        raise ValueError(f"Pair ID mismatch: en={pair_id}, tr={tr_record.get('pair_id')}")

    if en_record.get("split") != tr_record.get("split"):
        raise ValueError(f"Split mismatch for pair {pair_id}")

    if en_record.get("labels") != tr_record.get("labels"):
        raise ValueError(f"Labels mismatch for pair {pair_id}")
    if en_record.get("labels_direct") != tr_record.get("labels_direct"):
        raise ValueError(f"Direct label mismatch for pair {pair_id}")

    # Use aligned sentence pairs if available, otherwise synthesize alignment
    sentence_pairs = tr_record.get("sentence_alignment")
    if not sentence_pairs:
        from naltra.data.translation import segment_sentences

        en_sents = segment_sentences(en_record["text"])
        tr_sents = segment_sentences(tr_record["text"])
        min_len = min(len(en_sents), len(tr_sents))
        sentence_pairs = [
            {"index": i, "en": en_sents[i], "tr": tr_sents[i]} for i in range(min_len)
        ]

    mixed_text, primary_lang = mix_aligned_sentences(
        sentence_pairs=sentence_pairs,
        strategy=strategy,
        strength=strength,
        seed=base_seed,
        pair_id=pair_id,
    )

    pid = en_record.get("project_id", pair_id.replace("cordis:", ""))
    new_id = f"{pair_id}:codeswitch:{strategy}:{strength}"

    return {
        "id": new_id,
        "project_id": pid,
        "pair_id": pair_id,
        "original_pair_id": pair_id,
        "source_id": en_record.get("source_id", pid),
        "variant_of": en_record["id"],
        "title": en_record.get("title", ""),
        "text": mixed_text,
        "labels_direct": copy.deepcopy(en_record.get("labels_direct", en_record["labels"])),
        "labels": copy.deepcopy(en_record["labels"]),
        "language": "en-tr",
        "source": en_record.get("source", "cordis_h2020"),
        "license": en_record.get("license", "CC BY 4.0"),
        "split": en_record["split"],
        "taxonomy_version": en_record.get("taxonomy_version", "0.4.0"),
        "synthetic_language_variant": True,
        "en_id": en_record["id"],
        "tr_id": tr_record["id"],
        "code_switch_strategy": strategy,
        "code_switch_strength": strength,
        "primary_language": primary_lang,
        "base_seed": base_seed,
    }


def generate_code_switch_benchmarks(
    en_dir: str | Path | None = None,
    tr_dir: str | Path | None = None,
    output_base_dir: str | Path = "data/processed/cordis_h2020/code_switch",
    splits: Sequence[str] = ("validation", "test"),
    strategy: str = "sentence_mix",
    strength: str = "balanced",
    seed: int = 42,
    taxonomy_path: str | Path = "taxonomy/taxonomy.json",
    processed_base_dir: str | Path | None = None,
) -> dict[str, int]:
    """Generate synthetic EN/TR code-switched evaluation benchmarks."""
    if processed_base_dir is not None:
        base_p = Path(processed_base_dir)
        if not base_p.exists():
            raise FileNotFoundError(
                f"Cannot generate code-switch benchmarks: directory '{base_p}' does not exist."
            )
        if (base_p / "en").exists() and (base_p / "tr").exists():
            en_path = base_p / "en"
            tr_path = base_p / "tr"
            is_single_dir = False
        else:
            en_path = base_p
            tr_path = base_p
            is_single_dir = True
    else:
        en_path = Path(en_dir or "data/processed/cordis_h2020/en")
        tr_path = Path(tr_dir or "data/processed/cordis_h2020/tr")
        is_single_dir = False

    out_dir = Path(output_base_dir) / strategy / strength
    # The manifest records these relative to itself; refuse a cross-drive layout before writing.
    for path in (en_path, tr_path, *(p / f"{s}.jsonl" for p in (en_path, tr_path) for s in splits)):
        relative_path(path, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, int] = {}
    input_files = []

    for split_name in splits:
        if is_single_dir:
            split_file = en_path / f"{split_name}.jsonl"
            if not split_file.exists():
                raise FileNotFoundError(split_file)
            input_files.append(split_file)
            all_recs = load_jsonl(split_file)
            pairs = pair_aligned_records(all_recs)
        else:
            en_file = en_path / f"{split_name}.jsonl"
            tr_file = tr_path / f"{split_name}.jsonl"

            if not en_file.exists() or not tr_file.exists():
                raise FileNotFoundError(f"Missing EN/TR code-switch input for {split_name}.")
            input_files.extend([en_file, tr_file])

            en_recs = load_jsonl(en_file)
            tr_recs = load_jsonl(tr_file)
            pairs = pair_aligned_records(en_recs, tr_recs)

        mixed_records = [
            create_code_switched_record(
                en_rec,
                tr_rec,
                strategy=strategy,
                strength=strength,
                base_seed=seed,
            )
            for en_rec, tr_rec in pairs
        ]

        # Verify uniqueness
        ids = [r["id"] for r in mixed_records]
        if len(ids) != len(set(ids)):
            raise ValueError(
                f"Duplicate IDs found in generated {split_name} code-switched benchmark."
            )

        out_file = out_dir / f"{split_name}.jsonl"
        save_jsonl(mixed_records, out_file)
        results[split_name] = len(mixed_records)
        print(f"Generated {len(mixed_records)} code-switched records -> {out_file}")

    # Generate reproducibility manifest
    create_manifest(
        benchmark_name="cordis_h2020_code_switch",
        output_dir=out_dir,
        generation_parameters={
            "strategy": strategy,
            "strength": strength,
            "base_seed": seed,
            "splits": list(splits),
        },
        # Relative to the manifest, so the record does not depend on the checkout location.
        source_metadata={
            name: relative_path(path, out_dir)
            for name, path in (("source_en_dir", en_path), ("source_tr_dir", tr_path))
        },
        taxonomy_dir=Path(taxonomy_path).parent,
        input_files=input_files,
    )

    return results
