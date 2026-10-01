"""Small, dependency-free loaders for prepared NALTRA records."""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load non-empty JSON Lines records from a UTF-8 file."""
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Record on line {line_number} must be a JSON object.")
            records.append(record)
    return records


def save_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """Save records as JSON Lines into a UTF-8 file, creating parent directories if needed."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return target_path
