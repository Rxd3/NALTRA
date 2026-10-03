"""Deterministic synthetic EN/TR code-switch benchmark generation.

IMPORTANT: This module implements a controlled synthetic chunk-mixing robustness
benchmark. It preserves source-language token order within contiguous chunks, but
does NOT simulate linguistically natural, word-aligned, or grammatical code-switching.
English and Turkish spans are aligned at the document/record level from SIB-200,
not at the individual token or phrase translation level.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import random
from typing import Any, Sequence

from naltra.data.loader import load_jsonl, save_jsonl

VALID_CODE_SWITCH_STRATEGIES: set[str] = {
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
    records: Sequence[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Pair English and Turkish SIB-200 records by pair_id with strict validation."""
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for r in records:
        pid = r.get("pair_id")
        if not pid:
            raise ValueError(f"Record {r.get('id')} is missing 'pair_id'.")
        lang = r.get("language")
        if lang not in ("en", "tr"):
            continue
        if pid not in by_pair:
            by_pair[pid] = {}
        if lang in by_pair[pid]:
            raise ValueError(f"Duplicate {lang} record for pair_id '{pid}': {r.get('id')}")
        by_pair[pid][lang] = r

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for pid in sorted(by_pair.keys()):
        group = by_pair[pid]
        if "en" not in group or "tr" not in group:
            raise ValueError(f"Incomplete pair for pair_id '{pid}': found {list(group.keys())}")
        en_rec = group["en"]
        tr_rec = group["tr"]

        # Validate alignment
        if en_rec.get("split") != tr_rec.get("split"):
            raise ValueError(
                f"Split mismatch for pair_id '{pid}': en={en_rec.get('split')}, tr={tr_rec.get('split')}"
            )
        if en_rec.get("labels") != tr_rec.get("labels"):
            raise ValueError(
                f"Label mismatch for pair_id '{pid}': en={en_rec.get('labels')}, tr={tr_rec.get('labels')}"
            )
        pairs.append((en_rec, tr_rec))

    return pairs


def mix_code_switched_text(
    en_text: str,
    tr_text: str,
    strategy: str = "chunk_mix",
    strength: str = "balanced",
    seed: int = 42,
    pair_id: str = "sample",
) -> tuple[str, str]:
    """Generate synthetic mixed EN/TR text deterministically.

    Returns:
        tuple[str, str]: (mixed_text, primary_language)
    """
    if strategy not in VALID_CODE_SWITCH_STRATEGIES:
        raise ValueError(
            f"Unknown code-switch strategy '{strategy}'. Allowed: {sorted(VALID_CODE_SWITCH_STRATEGIES)}"
        )
    if strength not in VALID_CODE_SWITCH_STRENGTHS:
        raise ValueError(
            f"Unknown code-switch strength '{strength}'. Allowed: {sorted(VALID_CODE_SWITCH_STRENGTHS)}"
        )

    en_clean = en_text.strip()
    tr_clean = tr_text.strip()
    if not en_clean or not tr_clean:
        raise ValueError("Cannot code-switch empty or whitespace-only text.")

    en_tokens = en_clean.split()
    tr_tokens = tr_clean.split()

    exact_seed = get_code_switch_seed(pair_id, strategy, strength, base_seed=seed)
    rng = random.Random(exact_seed)

    # Deterministically choose primary language from stable per-pair seed
    # to avoid systematic English dominance
    primary_lang = "en" if rng.random() < 0.5 else "tr"
    if primary_lang == "en":
        primary_tokens, secondary_tokens = en_tokens, tr_tokens
    else:
        primary_tokens, secondary_tokens = tr_tokens, en_tokens

    len_p = len(primary_tokens)
    len_s = len(secondary_tokens)

    if strategy == "chunk_mix":
        if strength == "balanced":
            # Roughly 50/50 mix: first half from primary, second half from secondary
            cut_p = max(1, min(len_p - 1, int(round(len_p * 0.50)))) if len_p > 1 else 1
            cut_s = max(1, min(len_s - 1, int(round(len_s * 0.50)))) if len_s > 1 else 0
            mixed_tokens = primary_tokens[:cut_p] + secondary_tokens[cut_s:]
        elif strength == "light":
            # Roughly 75% primary, 25% secondary
            cut_p = max(1, min(len_p - 1, int(round(len_p * 0.75)))) if len_p > 1 else 1
            cut_s = max(1, min(len_s - 1, int(round(len_s * 0.75)))) if len_s > 1 else 0
            mixed_tokens = primary_tokens[:cut_p] + secondary_tokens[cut_s:]
        else:
            raise ValueError(f"Unhandled strength: {strength}")

    elif strategy == "interleave":
        # 3-span sandwich: primary head + secondary mid + primary tail
        if strength == "balanced":
            cut_p1 = max(1, len_p // 4) if len_p >= 4 else 1
            cut_p2 = max(cut_p1 + 1, (3 * len_p) // 4) if len_p >= 4 else len_p
            cut_s1 = max(0, len_s // 4) if len_s >= 4 else 0
            cut_s2 = max(cut_s1 + 1, (3 * len_s) // 4) if len_s >= 4 else len_s
            sec_span = secondary_tokens[cut_s1:cut_s2] if cut_s2 > cut_s1 else secondary_tokens
            mixed_tokens = primary_tokens[:cut_p1] + sec_span + primary_tokens[cut_p2:]
        elif strength == "light":
            cut_p1 = max(1, (3 * len_p) // 8) if len_p >= 4 else 1
            cut_p2 = max(cut_p1 + 1, (5 * len_p) // 8) if len_p >= 4 else len_p
            cut_s1 = max(0, (3 * len_s) // 8) if len_s >= 4 else 0
            cut_s2 = max(cut_s1 + 1, (5 * len_s) // 8) if len_s >= 4 else len_s
            sec_span = secondary_tokens[cut_s1:cut_s2] if cut_s2 > cut_s1 else secondary_tokens[:1]
            mixed_tokens = primary_tokens[:cut_p1] + sec_span + primary_tokens[cut_p2:]
        else:
            raise ValueError(f"Unhandled strength: {strength}")
    else:
        raise ValueError(f"Unhandled strategy: {strategy}")

    # Fallback to ensure both languages contribute if both texts had >= 2 tokens
    if len_p >= 2 and len_s >= 2:
        sec_set = set(secondary_tokens)
        if not any(t in sec_set for t in mixed_tokens):
            mixed_tokens.append(secondary_tokens[-1])

    result = " ".join(mixed_tokens).strip()
    return result, primary_lang


def create_code_switched_record(
    en_record: dict[str, Any],
    tr_record: dict[str, Any],
    strategy: str = "chunk_mix",
    strength: str = "balanced",
    base_seed: int = 42,
) -> dict[str, Any]:
    """Create a synthetic code-switched record from an aligned EN/TR pair."""
    pair_id = en_record.get("pair_id")
    if not pair_id or pair_id != tr_record.get("pair_id"):
        raise ValueError(f"Pair ID mismatch: en={pair_id}, tr={tr_record.get('pair_id')}")

    if en_record.get("split") != tr_record.get("split"):
        raise ValueError(f"Split mismatch for pair {pair_id}")

    if en_record.get("labels") != tr_record.get("labels"):
        raise ValueError(f"Labels mismatch for pair {pair_id}")

    if en_record.get("language") != "en":
        raise ValueError(f"en_record must have language 'en', got '{en_record.get('language')}'")
    if tr_record.get("language") != "tr":
        raise ValueError(f"tr_record must have language 'tr', got '{tr_record.get('language')}'")

    mixed_text, primary_lang = mix_code_switched_text(
        en_text=en_record["text"],
        tr_text=tr_record["text"],
        strategy=strategy,
        strength=strength,
        seed=base_seed,
        pair_id=pair_id,
    )

    new_id = f"{pair_id}:codeswitch:{strategy}:{strength}"

    return {
        "id": new_id,
        "text": mixed_text,
        "labels": copy.deepcopy(en_record["labels"]),
        "language": "en-tr",
        "source": "sib200",
        "license": en_record.get("license", "CC BY-SA 4.0"),
        "split": en_record["split"],
        "pair_id": pair_id,
        "en_id": en_record["id"],
        "tr_id": tr_record["id"],
        "code_switch_strategy": strategy,
        "code_switch_strength": strength,
        "primary_language": primary_lang,
        "original_pair_id": pair_id,
    }


def generate_code_switch_benchmarks(
    processed_base_dir: str | Path = "data/processed/sib200",
    output_base_dir: str | Path = "data/processed/code_switch",
    splits: Sequence[str] = ("validation", "test"),
    strategy: str = "chunk_mix",
    strength: str = "balanced",
    seed: int = 42,
) -> dict[str, int]:
    """Generate synthetic EN/TR code-switched validation/test benchmark copies."""
    in_dir = Path(processed_base_dir)
    out_dir = Path(output_base_dir) / strategy / strength
    results: dict[str, int] = {}

    for split_name in splits:
        input_file = in_dir / f"{split_name}.jsonl"
        if not input_file.exists():
            print(f"[WARN] Input file missing, skipping: {input_file}")
            continue

        raw_records = load_jsonl(input_file)
        pairs = pair_aligned_records(raw_records)

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

        # Verify unique IDs
        ids = [r["id"] for r in mixed_records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate IDs found in generated {split_name} code-switched benchmark.")

        output_file = out_dir / f"{split_name}.jsonl"
        save_jsonl(mixed_records, output_file)
        results[split_name] = len(mixed_records)
        print(f"Generated {len(mixed_records)} code-switched records -> {output_file}")

    return results
