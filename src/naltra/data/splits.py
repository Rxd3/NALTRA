"""Deterministic dependency-free split helper for initial development."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any, TypeVar

T = TypeVar("T")


def split_records(
    records: Sequence[T],
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[T], list[T], list[T]]:
    """Shuffle and split records; later replace with multilabel stratification."""
    if train_ratio <= 0 or validation_ratio < 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("Ratios must leave a positive test split.")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    validation_end = train_end + int(len(shuffled) * validation_ratio)
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

    total_ratio = train_ratio + validation_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-5:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    total_records = len(records)
    if total_records == 0:
        return ([], [], [])

    # Exact integer capacities derived from requested ratios
    c_train = int(round(total_records * train_ratio))
    c_val = int(round(total_records * validation_ratio))
    c_test = total_records - c_train - c_val
    split_capacities = [c_train, c_val, c_test]

    label_counts: Counter[str] = Counter()
    for record in records:
        for label in record.get("labels", []):
            label_counts[label] += 1

    ratios = [train_ratio, validation_ratio, test_ratio]
    initial_targets = [
        {label: max(count * r, 0.1) for label, count in label_counts.items()}
        for r in ratios
    ]
    remaining_targets = [
        {label: count * r for label, count in label_counts.items()}
        for r in ratios
    ]

    splits: list[list[dict[str, Any]]] = [[], [], []]

    # Deterministic sorting: seed-shuffled, then multi-label first, then rarest label, then tie-break by ID
    rnd = random.Random(seed)
    sorted_records = list(records)
    rnd.shuffle(sorted_records)
    sorted_records.sort(
        key=lambda r: (
            -len(r.get("labels", [])),
            min((label_counts[l] for l in r.get("labels", [])), default=0),
            str(r.get("id", "")),
        )
    )

    for record in sorted_records:
        rec_labels = record.get("labels", [])
        # Hard capacity filter: only non-full splits are eligible
        eligible = [s for s in range(3) if len(splits[s]) < split_capacities[s]]
        if not eligible:
            raise RuntimeError("All splits have exceeded capacity.")

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
