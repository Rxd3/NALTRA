"""Deterministic, label-preserving text perturbations for NALTRA robustness benchmarking."""

from __future__ import annotations

import copy
import hashlib
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from naltra.data.loader import load_jsonl, save_jsonl

SEVERITY_LEVELS: dict[str, float] = {
    "light": 0.05,
    "medium": 0.10,
}

VALID_STRATEGIES: set[str] = {
    "char_swap",
    "char_delete",
    "char_duplicate",
    "whitespace",
    "capitalization",
    "punctuation",
    "turkish_diacritics",
    "combined",
}

TURKISH_DIACRITICS_MAP: dict[str, str] = {
    "ç": "c",
    "Ç": "C",
    "ğ": "g",
    "Ğ": "G",
    "ı": "i",
    "İ": "I",
    "ö": "o",
    "Ö": "O",
    "ş": "s",
    "Ş": "S",
    "ü": "u",
    "Ü": "U",
}


def get_deterministic_seed(
    record_id: str,
    strategy: str,
    severity: str | float,
    base_seed: int = 42,
) -> int:
    """Generate a platform-independent integer seed via SHA-256."""
    key = f"{record_id}:{strategy}:{severity}:{base_seed}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big")


def _resolve_severity(severity: str | float) -> float:
    if isinstance(severity, str):
        if severity not in SEVERITY_LEVELS:
            raise ValueError(
                f"Unknown severity level '{severity}'. Allowed: {sorted(SEVERITY_LEVELS)}"
            )
        return SEVERITY_LEVELS[severity]
    if isinstance(severity, (int, float)):
        prob = float(severity)
        if not (0.0 < prob <= 1.0):
            raise ValueError(f"Numeric severity must be in range (0.0, 1.0], got {prob}")
        return prob
    raise TypeError(f"Severity must be string or float, got {type(severity).__name__}")


def _perturb_char_swap(text: str, prob: float, rng: random.Random) -> str:
    words = text.split(" ")
    new_words = []
    edited = False
    for word in words:
        swap_candidates = [j for j in range(len(word) - 1) if word[j] != word[j + 1]]
        if swap_candidates and rng.random() < prob:
            idx = rng.choice(swap_candidates)
            chars = list(word)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            new_words.append("".join(chars))
            edited = True
        else:
            new_words.append(word)

    # Fallback: force at least one swap if possible and no edits occurred
    if not edited and len(words) > 0:
        candidates = [
            i for i, w in enumerate(new_words) if any(w[j] != w[j + 1] for j in range(len(w) - 1))
        ]
        if candidates:
            target_idx = rng.choice(candidates)
            w = new_words[target_idx]
            swap_candidates = [j for j in range(len(w) - 1) if w[j] != w[j + 1]]
            idx = rng.choice(swap_candidates)
            chars = list(w)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            new_words[target_idx] = "".join(chars)
        else:
            # If all words consist of identical repeated chars or single chars
            return _perturb_char_duplicate(text, prob=1.0, rng=rng)

    return " ".join(new_words)


def _perturb_char_delete(text: str, prob: float, rng: random.Random) -> str:
    chars = list(text)
    if not chars:
        return text
    new_chars = []
    for c in chars:
        if c.isalnum() and rng.random() < prob:
            continue
        new_chars.append(c)

    # Fallback: force at least one deletion of alphanumeric character
    if len(new_chars) == len(chars):
        alnum_indices = [i for i, c in enumerate(chars) if c.isalnum()]
        if len(alnum_indices) > 1:
            del_idx = rng.choice(alnum_indices)
            new_chars = [c for i, c in enumerate(chars) if i != del_idx]
        elif len(alnum_indices) == 1:
            # Single char: duplicate instead to avoid returning empty text
            return _perturb_char_duplicate(text, prob=1.0, rng=rng)

    result = "".join(new_chars).strip()
    return result if result else text


def _perturb_char_duplicate(text: str, prob: float, rng: random.Random) -> str:
    chars = list(text)
    if not chars:
        return text
    new_chars = []
    for c in chars:
        new_chars.append(c)
        if c.isalnum() and rng.random() < prob:
            new_chars.append(c)

    # Fallback: force at least one duplication
    if len(new_chars) == len(chars):
        alnum_indices = [i for i, c in enumerate(chars) if c.isalnum()]
        if alnum_indices:
            dup_idx = rng.choice(alnum_indices)
            new_chars.insert(dup_idx, chars[dup_idx])
        else:
            # Non-alphanumeric text
            new_chars.append(chars[-1])

    return "".join(new_chars)


def _perturb_whitespace(text: str, prob: float, rng: random.Random) -> str:
    words = text.split(" ")
    if len(words) <= 1:
        if len(text) > 1:
            mid = len(text) // 2
            return text[:mid] + " " + text[mid:]
        # Single character cannot have interior whitespace; duplicate as safe fallback
        return _perturb_char_duplicate(text, prob=1.0, rng=rng)
    new_tokens = []
    edited = False
    for w in words[:-1]:
        new_tokens.append(w)
        # Random extra space
        if rng.random() < prob:
            new_tokens.append("")
            edited = True

    new_tokens.append(words[-1])
    res = " ".join(new_tokens)
    if not edited:
        # Fallback: force double space between first two words
        res = words[0] + "  " + " ".join(words[1:])
    return res


def _perturb_capitalization(text: str, prob: float, rng: random.Random) -> str:
    chars = list(text)
    new_chars = []
    for c in chars:
        if c.isalpha() and rng.random() < prob:
            new_chars.append(c.lower() if c.isupper() else c.upper())
        else:
            new_chars.append(c)

    # Fallback: toggle at least one letter
    if new_chars == chars:
        alpha_indices = [i for i, c in enumerate(chars) if c.isalpha()]
        if alpha_indices:
            idx = rng.choice(alpha_indices)
            c = chars[idx]
            new_chars[idx] = c.lower() if c.isupper() else c.upper()
        else:
            return _perturb_char_duplicate(text, prob=1.0, rng=rng)

    return "".join(new_chars)


def _perturb_punctuation(text: str, prob: float, rng: random.Random) -> str:
    punct_map = {".": ",", ",": ".", "!": "?", "?": "!", ";": ":", ":": ";"}
    chars = list(text)
    new_chars = []
    for c in chars:
        if c in punct_map and rng.random() < prob:
            new_chars.append(punct_map[c])
        else:
            new_chars.append(c)

    # Fallback: replace or append a punctuation mark
    if new_chars == chars:
        p_indices = [i for i, c in enumerate(chars) if c in punct_map]
        if p_indices:
            idx = rng.choice(p_indices)
            new_chars[idx] = punct_map[chars[idx]]
        else:
            new_chars.append(".")

    return "".join(new_chars)


def _perturb_turkish_diacritics(text: str, prob: float, rng: random.Random) -> str:
    chars = list(text)
    new_chars = []
    for c in chars:
        if c in TURKISH_DIACRITICS_MAP and rng.random() < prob:
            new_chars.append(TURKISH_DIACRITICS_MAP[c])
        else:
            new_chars.append(c)

    # Fallback: replace at least one Turkish diacritic if present
    if new_chars == chars:
        tr_indices = [i for i, c in enumerate(chars) if c in TURKISH_DIACRITICS_MAP]
        if tr_indices:
            idx = rng.choice(tr_indices)
            new_chars[idx] = TURKISH_DIACRITICS_MAP[chars[idx]]
        else:
            # If no Turkish diacritics exist in this specific Turkish string, perform character swap
            return _perturb_char_swap(text, prob=0.1, rng=rng)

    return "".join(new_chars)


def _perturb_combined(text: str, prob: float, rng: random.Random) -> str:
    """Realistic composite perturbation for benchmark robustness evaluation.

    Contains:
    1. Primary typographical character perturbation:
       - 35% probability: adjacent character swap (typo / slip)
       - 35% probability: character duplication (stuck key / stutter)
       - 30% probability: character deletion (dropped keystroke)
    2. Secondary perturbations:
       - Capitalization variation (casing flip on letters)
       - Punctuation variation (swapping equivalent or nearby punctuation marks)
    3. Minimum-edit guarantee:
       - Deterministic fallback ensures at least 1 edit occurs and text is never empty.
    """
    dice = rng.random()
    if dice < 0.35:
        res = _perturb_char_swap(text, prob, rng)
    elif dice < 0.70:
        res = _perturb_char_duplicate(text, prob, rng)
    else:
        res = _perturb_char_delete(text, prob, rng)

    # Subtle capitalization or punctuation change
    if rng.random() < prob:
        res = _perturb_capitalization(res, prob=0.05, rng=rng)
    if rng.random() < prob:
        res = _perturb_punctuation(res, prob=0.10, rng=rng)

    # Guarantee text has changed
    if res == text:
        res = _perturb_char_duplicate(text, prob=1.0, rng=rng)
        if res == text:
            res = _perturb_capitalization(text, prob=1.0, rng=rng)

    return res


def perturb_text(
    text: str,
    strategy: str = "combined",
    severity: str | float = "medium",
    seed: int = 42,
    language: str = "en",
    record_id: str = "sample",
) -> str:
    """Create a controlled, deterministic text perturbation."""
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Unknown noise strategy '{strategy}'. Allowed: {sorted(VALID_STRATEGIES)}"
        )

    prob = _resolve_severity(severity)
    exact_seed = get_deterministic_seed(record_id, strategy, severity, base_seed=seed)
    rng = random.Random(exact_seed)

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Cannot perturb empty or non-string text.")

    if strategy == "turkish_diacritics":
        if language != "tr":
            raise ValueError(
                f"Strategy 'turkish_diacritics' is only valid for language 'tr', got '{language}'."
            )
        noisy = _perturb_turkish_diacritics(text, prob, rng)
    elif strategy == "char_swap":
        noisy = _perturb_char_swap(text, prob, rng)
    elif strategy == "char_delete":
        noisy = _perturb_char_delete(text, prob, rng)
    elif strategy == "char_duplicate":
        noisy = _perturb_char_duplicate(text, prob, rng)
    elif strategy == "whitespace":
        noisy = _perturb_whitespace(text, prob, rng)
    elif strategy == "capitalization":
        noisy = _perturb_capitalization(text, prob, rng)
    elif strategy == "punctuation":
        noisy = _perturb_punctuation(text, prob, rng)
    elif strategy == "combined":
        noisy = _perturb_combined(text, prob, rng)
    clean_orig = text.strip()
    result = noisy.strip()

    # Enforce minimum-edit guarantee after final output normalization/strip
    if not result or result == clean_orig:
        res = _perturb_char_duplicate(clean_orig, prob=1.0, rng=rng).strip()
        if not res or res == clean_orig:
            res = _perturb_capitalization(clean_orig, prob=1.0, rng=rng).strip()
        if not res or res == clean_orig:
            res = clean_orig + (clean_orig[-1] if clean_orig else "x")
        result = res

    return result


def create_noisy_record(
    record: dict[str, Any],
    strategy: str = "combined",
    severity: str | float = "medium",
    base_seed: int = 42,
) -> dict[str, Any]:
    """Return a new deep-copied record with perturbed text and traceability metadata."""
    if not isinstance(record, dict) or "id" not in record or "text" not in record:
        raise ValueError("Record must be a dictionary containing 'id' and 'text'.")

    lang = record.get("language", "en")
    record_id = record["id"]

    noisy_text = perturb_text(
        text=record["text"],
        strategy=strategy,
        severity=severity,
        seed=base_seed,
        language=lang,
        record_id=record_id,
    )

    # Independent deep copy to protect all mutable metadata
    noisy_record = copy.deepcopy(record)
    noisy_record["id"] = f"{record_id}:noise:{strategy}:{severity}"
    noisy_record["text"] = noisy_text
    noisy_record["original_id"] = record_id
    noisy_record["noise_strategy"] = strategy
    noisy_record["noise_severity"] = str(severity)
    noisy_record["base_seed"] = base_seed

    return noisy_record


def generate_noisy_benchmarks(
    processed_base_dir: str | Path = "data/processed",
    output_base_dir: str | Path = "data/noisy",
    datasets: Sequence[str] = ("sib200", "multifin", "mn_ds"),
    splits: Sequence[str] = ("validation", "test"),
    strategy: str = "combined",
    severity: str = "medium",
    seed: int = 42,
) -> dict[str, dict[str, int]]:
    """Generate noisy validation/test benchmark copies from existing clean processed files."""
    from naltra.data.manifest import create_manifest

    proc_base = Path(processed_base_dir)
    out_base = Path(output_base_dir) / strategy / str(severity)

    # Preflight check: all required clean files must exist
    missing_files: list[str] = []
    for dataset_name in datasets:
        for split_name in splits:
            input_file = proc_base / dataset_name / f"{split_name}.jsonl"
            if not input_file.exists():
                missing_files.append(str(input_file))

    if missing_files:
        files_str = "\n  - ".join(missing_files)
        raise FileNotFoundError(
            f"Cannot generate noisy benchmarks because required clean input files are missing:\n"
            f"  - {files_str}\n"
            f"Please run clean dataset preparation first:\n"
            f"  python scripts/prepare_data.py --dataset all"
        )

    results: dict[str, dict[str, int]] = {}

    for dataset_name in datasets:
        results[dataset_name] = {}
        dataset_out_dir = out_base / dataset_name
        dataset_out_dir.mkdir(parents=True, exist_ok=True)

        for split_name in splits:
            input_file = proc_base / dataset_name / f"{split_name}.jsonl"
            clean_records = load_jsonl(input_file)
            noisy_records = [
                create_noisy_record(r, strategy=strategy, severity=severity, base_seed=seed)
                for r in clean_records
            ]

            output_file = dataset_out_dir / f"{split_name}.jsonl"
            save_jsonl(noisy_records, output_file)
            results[dataset_name][split_name] = len(noisy_records)
            print(f"Generated {len(noisy_records)} noisy records -> {output_file}")

        # Generate reproducibility manifest for this noisy benchmark dataset
        create_manifest(
            benchmark_name=f"noisy_{dataset_name}",
            output_dir=dataset_out_dir,
            generation_parameters={
                "strategy": strategy,
                "severity": severity,
                "base_seed": seed,
                "splits": list(splits),
            },
            source_metadata={
                "source_dataset": dataset_name,
                "processed_base_dir": str(proc_base),
            },
        )

    # Top-level manifest for the entire noisy benchmark configuration
    create_manifest(
        benchmark_name="noisy_robustness_benchmark",
        output_dir=out_base,
        generation_parameters={
            "strategy": strategy,
            "severity": severity,
            "base_seed": seed,
            "datasets": list(datasets),
            "splits": list(splits),
        },
        source_metadata={
            "processed_base_dir": str(proc_base),
        },
    )

    return results
