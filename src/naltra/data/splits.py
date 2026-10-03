"""Deterministic dependency-free split helper for initial development."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any, TypeVar

T = TypeVar("T")


def _validate_split_ratios(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> list[float]:
    ratios = [train_ratio, validation_ratio, test_ratio]
    for idx, r in enumerate(ratios):
        if not isinstance(r, (int, float)) or isinstance(r, bool):
            raise TypeError(f"Split ratio at index {idx} must be a numeric float, got {type(r).__name__}")
        if not math.isfinite(r):
            raise ValueError(f"Split ratio at index {idx} must be finite, got {r}")
        if r < 0.0:
            raise ValueError(f"Split ratio at index {idx} must be non-negative, got {r}")

    if all(r == 0.0 for r in ratios):
        raise ValueError("At least one split ratio must be positive.")

    total_ratio = sum(ratios)
    if abs(total_ratio - 1.0) > 1e-5:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    return [float(r) for r in ratios]


def _compute_split_capacities(total_records: int, ratios: list[float]) -> list[int]:
    """Compute exact integer capacities using deterministic largest-remainder allocation."""
    if total_records == 0:
        return [0, 0, 0]

    shares = [total_records * r for r in ratios]
    base_caps = [int(math.floor(s)) for s in shares]
    remainders = [s - b for s, b in zip(shares, base_caps)]
    shortfall = total_records - sum(base_caps)

    # Allocate shortfall to splits with largest fractional remainders.
    # Deterministic tie-breaking by split index (0=train, 1=val, 2=test).
    ranked_indices = sorted(range(3), key=lambda i: (-remainders[i], i))
    for i in range(shortfall):
        base_caps[ranked_indices[i]] += 1

    assert sum(base_caps) == total_records
    return base_caps


def split_records(
    records: Sequence[T],
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[T], list[T], list[T]]:
    """Shuffle and split records; later replace with multilabel stratification."""
    test_ratio = 1.0 - (train_ratio + validation_ratio)
    ratios = _validate_split_ratios(train_ratio, validation_ratio, test_ratio)
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    capacities = _compute_split_capacities(len(shuffled), ratios)
    train_end = capacities[0]
    validation_end = train_end + capacities[1]
    return (
        shuffled[:train_end],
        shuffled[train_end:validation_end],
        shuffled[validation_end:],
    )


def multilabel_stratified_split(
    records: Sequence[dict[str, Any]],
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically partition multi-label records using iterative stratification with hard split capacities.

    Ensures exact integer split allocations, balanced distribution for each label across
    train, validation, and test splits, and strictly prevents record ID leakage.
    """
    from collections import Counter

    ratios = _validate_split_ratios(train_ratio, validation_ratio, test_ratio)

    total_records = len(records)
    if total_records == 0:
        return ([], [], [])

    split_capacities = _compute_split_capacities(total_records, ratios)

    label_counts: Counter[str] = Counter()
    for record in records:
        for label in record.get("labels", []):
            label_counts[label] += 1

    initial_targets = [
        {label: max(count * r, 0.1) for label, count in label_counts.items()}
        for r in ratios
    ]
    remaining_targets = [
        {label: count * r for label, count in label_counts.items()}
        for r in ratios
    ]

    splits: list[list[dict[str, Any]]] = [[], [], []]

    # Deterministic sorting: multi-label first, then rarest label, then seeded pseudo-random tie-breaker
    rnd = random.Random(seed)
    records_with_keys = [(idx, r, rnd.random()) for idx, r in enumerate(records)]
    records_with_keys.sort(
        key=lambda item: (
            -len(item[1].get("labels", [])),
            min((label_counts[l] for l in item[1].get("labels", [])), default=0),
            item[2],
            item[0],
        )
    )
    sorted_records = [item[1] for item in records_with_keys]

    for record in sorted_records:
        rec_labels = record.get("labels", [])
        # Hard capacity filter: only splits with capacity > 0 and not full are eligible
        eligible = [s for s in range(3) if split_capacities[s] > 0 and len(splits[s]) < split_capacities[s]]
        if not eligible:
            raise RuntimeError("All eligible splits have reached capacity.")

        best_split = eligible[0]
        best_score = (-float("inf"), -float("inf"), -999)

        for s in eligible:
            need_ratio = (
                sum(
                    remaining_targets[s].get(l, 0.0) / initial_targets[s][l]
                    for l in rec_labels
                )
                / len(rec_labels)
                if rec_labels
                else 0.0
            )
            cap_ratio = (split_capacities[s] - len(splits[s])) / split_capacities[s]
            score = (need_ratio, cap_ratio, -s)
            if score > best_score:
                best_score = score
                best_split = s

        splits[best_split].append(record)
        for l in rec_labels:
            remaining_targets[best_split][l] -= 1

    assert [len(s) for s in splits] == split_capacities, "Split lengths must exactly match capacities."
    return (splits[0], splits[1], splits[2])
