"""Split utilities for stratified and reproducible dataset partitioning."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

T = TypeVar("T")


def _validate_split_ratios(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> list[float]:
    """Validate that split ratios are finite, non-negative, and sum to 1.0."""
    ratios = [train_ratio, validation_ratio, test_ratio]

    for r in ratios:
        if isinstance(r, bool) or not isinstance(r, (int, float)):
            raise TypeError(f"Split ratio must be a numeric float, got {type(r).__name__}: {r!r}")
        if not math.isfinite(r):
            raise ValueError(f"Split ratio must be finite, got: {r}")
        if r < 0.0:
            raise ValueError(f"Split ratio must be non-negative, got: {r}")

    if all(r == 0.0 for r in ratios):
        raise ValueError("At least one split ratio must be positive.")

    total = sum(ratios)
    if not math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError(
            f"Split ratios must sum to 1.0, got "
            f"{train_ratio} + {validation_ratio} + {test_ratio} = {total}"
        )

    return [float(r) for r in ratios]


def _compute_split_capacities(total_records: int, ratios: list[float]) -> list[int]:
    """Compute exact integer capacities using deterministic largest-remainder allocation."""
    if total_records == 0:
        return [0, 0, 0]

    shares = [total_records * r for r in ratios]
    base_caps = [int(math.floor(s)) for s in shares]
    remainders = [s - b for s, b in zip(shares, base_caps, strict=True)]
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

    c_train, c_val, c_test = capacities
    train = shuffled[:c_train]
    val = shuffled[c_train : c_train + c_val]
    test = shuffled[c_train + c_val :]

    return train, val, test


def multilabel_stratified_split(
    records: Sequence[dict[str, Any]],
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically partition multi-label records using iterative stratification.

    Ensures exact integer split allocations, balanced distribution for each label across
    train, validation, and test splits, and strictly prevents record ID leakage.
    """
    from collections import Counter

    ratios = _validate_split_ratios(train_ratio, validation_ratio, test_ratio)

    total_records = len(records)
    if total_records == 0:
        return ([], [], [])

    # Reject duplicate non-empty source record IDs
    seen_ids: set[str] = set()
    for record in records:
        rec_id = record.get("id")
        if rec_id is not None and isinstance(rec_id, str) and rec_id.strip():
            if rec_id in seen_ids:
                raise ValueError(f"Duplicate record ID found before splitting: '{rec_id}'")
            seen_ids.add(rec_id)

    split_capacities = _compute_split_capacities(total_records, ratios)

    label_counts: Counter[str] = Counter()
    for record in records:
        for label in record.get("labels", []):
            label_counts[label] += 1

    initial_targets = [
        {label: max(count * r, 0.1) for label, count in label_counts.items()} for r in ratios
    ]
    remaining_targets = [
        {label: count * r for label, count in label_counts.items()} for r in ratios
    ]

    splits: list[list[dict[str, Any]]] = [[], [], []]

    # Deterministic sorting: multi-label first, then rarest label, then seeded tie-breaker
    rnd = random.Random(seed)
    records_with_keys = [(idx, r, rnd.random()) for idx, r in enumerate(records)]
    records_with_keys.sort(
        key=lambda item: (
            -len(item[1].get("labels", [])),
            min((label_counts[lbl] for lbl in item[1].get("labels", [])), default=0),
            item[2],
            item[0],
        )
    )
    sorted_records = [item[1] for item in records_with_keys]

    for record in sorted_records:
        rec_labels = record.get("labels", [])
        eligible = [
            s for s in range(3) if split_capacities[s] > 0 and len(splits[s]) < split_capacities[s]
        ]
        if not eligible:
            raise RuntimeError("All eligible splits have reached capacity.")

        best_split = eligible[0]
        best_score = (-float("inf"), -float("inf"), -999)

        for s in eligible:
            need_ratio = (
                sum(
                    remaining_targets[s].get(lbl, 0.0) / initial_targets[s][lbl]
                    for lbl in rec_labels
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
        for lbl in rec_labels:
            remaining_targets[best_split][lbl] -= 1

    assert [
        len(s) for s in splits
    ] == split_capacities, "Split lengths must exactly match capacities."
    return (splits[0], splits[1], splits[2])


def grouped_multilabel_stratified_split(
    records: Sequence[dict[str, Any]],
    group_key: str | Callable[[dict[str, Any]], str],
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically partition multi-label records grouped by atomic keys.

    Ensures that all records sharing the same group key are assigned to the exact same split,
    preventing any train/evaluation leakage, while optimizing multi-label balance and split sizes.
    """
    from collections import Counter, defaultdict

    ratios = _validate_split_ratios(train_ratio, validation_ratio, test_ratio)

    total_records = len(records)
    if total_records == 0:
        return ([], [], [])

    # Reject duplicate non-empty source record IDs
    seen_ids: set[str] = set()
    for record in records:
        rec_id = record.get("id")
        if rec_id is not None and isinstance(rec_id, str) and rec_id.strip():
            if rec_id in seen_ids:
                raise ValueError(f"Duplicate record ID found before splitting: '{rec_id}'")
            seen_ids.add(rec_id)

    # Group records by key
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = group_key(record) if callable(group_key) else str(record.get(group_key, ""))
        groups[key].append(record)

    group_items = []
    for g_id, rec_list in groups.items():
        combined_labels = sorted(list({lbl for r in rec_list for lbl in r.get("labels", [])}))
        group_items.append(
            {
                "group_id": g_id,
                "records": rec_list,
                "size": len(rec_list),
                "labels": combined_labels,
            }
        )

    capacities = _compute_split_capacities(total_records, ratios)

    label_counts: Counter[str] = Counter()
    for g in group_items:
        for lbl in g["labels"]:
            label_counts[lbl] += g["size"]

    initial_targets = [
        {lbl: max(count * r, 0.1) for lbl, count in label_counts.items()} for r in ratios
    ]
    remaining_targets = [{lbl: count * r for lbl, count in label_counts.items()} for r in ratios]

    split_groups: list[list[dict[str, Any]]] = [[], [], []]
    split_rec_counts = [0, 0, 0]

    rnd = random.Random(seed)
    items_with_keys = [(idx, g, rnd.random()) for idx, g in enumerate(group_items)]
    items_with_keys.sort(
        key=lambda item: (
            -len(item[1]["labels"]),
            min((label_counts[lbl] for lbl in item[1]["labels"]), default=0),
            item[2],
            item[0],
        )
    )
    sorted_groups = [item[1] for item in items_with_keys]

    for g in sorted_groups:
        g_size = g["size"]
        rec_labels = g["labels"]

        eligible = [
            s
            for s in range(3)
            if capacities[s] > 0 and split_rec_counts[s] + g_size <= capacities[s]
        ]
        if not eligible:
            eligible = [
                s for s in range(3) if capacities[s] > 0 and split_rec_counts[s] < capacities[s]
            ]
        if not eligible:
            eligible = [s for s in range(3) if capacities[s] > 0]
        if not eligible:
            eligible = [0]

        best_split = eligible[0]
        best_score = (-float("inf"), -float("inf"), -999)

        for s in eligible:
            need_ratio = (
                sum(
                    remaining_targets[s].get(lbl, 0.0) / initial_targets[s][lbl]
                    for lbl in rec_labels
                )
                / len(rec_labels)
                if rec_labels
                else 0.0
            )
            cap_ratio = (
                (capacities[s] - split_rec_counts[s]) / capacities[s] if capacities[s] > 0 else 0.0
            )
            score = (need_ratio, cap_ratio, -s)
            if score > best_score:
                best_score = score
                best_split = s

        split_groups[best_split].append(g)
        split_rec_counts[best_split] += g_size
        for lbl in rec_labels:
            remaining_targets[best_split][lbl] -= g_size

    splits = [[r for g in split_groups[s] for r in g["records"]] for s in range(3)]
    assert sum(len(s) for s in splits) == total_records, "All records must be assigned."
    return (splits[0], splits[1], splits[2])
