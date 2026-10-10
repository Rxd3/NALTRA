"""Resume a copied Laya validation cache on Windows CUDA with PyTorch 2.7.1/cu128."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def compatible_identity(identity: dict[str, Any]) -> dict[str, Any]:
    """Allow the documented CUDA runtime change, retaining every scientific input."""
    value = copy.deepcopy(identity)
    environment = value["environment"]
    if (
        value["device"] != "cuda"
        or value["config"]["model"] != "laya"
        or environment["platform"] != "win32"
        or not re.fullmatch(r"3\.13\.\d+", environment["python_version"])
        or environment["package_versions"]["torch"] not in {"2.6.0+cu124", "2.7.1+cu128"}
    ):
        raise ValueError("Transfer supports Windows, Python 3.13 and the pinned CUDA builds only.")
    value.pop("hardware")
    environment.pop("python_version")
    environment["package_versions"].pop("torch")
    return value


def transfer_identity(path: Path, key: str, identity: dict[str, Any]) -> None:
    """Record the original runtime before rebinding a copied cache; preserve all scores."""
    if not path.is_file():
        raise ValueError("Copy prediction_cache.sqlite3 into the output directory before resuming.")
    connection = sqlite3.connect(path)
    try:
        compatible_identity(identity)
        digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if connection.execute(
            "SELECT name FROM sqlite_master WHERE name = 'runtime_transfers'"
        ).fetchone() and any(
            row[0] != digest
            for row in connection.execute("SELECT adapter_sha256 FROM runtime_transfers")
        ):
            raise ValueError("The transfer adapter changed; retain its original version.")
        previous = connection.execute("SELECT identity FROM runs WHERE key = ?", (key,)).fetchone()
        if previous is None:
            return  # This language has not started yet; the ordinary cache creates its run.
        source = json.loads(previous[0])
        if source == identity:
            return
        if compatible_identity(source) != compatible_identity(identity):
            raise ValueError(
                "Transfer provenance differs: retain the exact model, data, code, settings "
                "and pinned dependencies. Extract the transfer ZIP over the cloned repository."
            )
        with connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS runtime_transfers ("
                "id INTEGER PRIMARY KEY, run TEXT NOT NULL, completed_records INTEGER NOT NULL, "
                "source_identity TEXT NOT NULL, target_identity TEXT NOT NULL, "
                "adapter_sha256 TEXT NOT NULL, created_at_utc TEXT NOT NULL)"
            )
            count = connection.execute(
                "SELECT COUNT(*) FROM predictions WHERE run = ?", (key,)
            ).fetchone()[0]
            encoded = json.dumps(identity, sort_keys=True, allow_nan=False)
            connection.execute(
                "INSERT INTO runtime_transfers "
                "(run, completed_records, source_identity, target_identity, "
                "adapter_sha256, created_at_utc) VALUES (?, ?, ?, ?, ?, ?)",
                (key, count, previous[0], encoded, digest, datetime.now(UTC).isoformat()),
            )
            connection.execute("UPDATE runs SET identity = ? WHERE key = ?", (encoded, key))
    finally:
        connection.close()


def annotate_report(output: Path) -> None:
    """Make mixed-runtime provenance visible; pooled timing is not a 5070 benchmark."""
    database = output / "prediction_cache.sqlite3"
    report = output / "evaluation_summary.json"
    if not report.is_file() or not database.is_file():
        return
    connection = sqlite3.connect(database)
    try:
        if not connection.execute(
            "SELECT name FROM sqlite_master WHERE name = 'runtime_transfers'"
        ).fetchone():
            return
        history = [
            {
                "track": row[0],
                "records_before_transfer": row[1],
                "source_identity": json.loads(row[2]),
                "target_identity": json.loads(row[3]),
                "adapter_sha256": row[4],
                "created_at_utc": row[5],
            }
            for row in connection.execute(
                "SELECT run, completed_records, source_identity, target_identity, "
                "adapter_sha256, created_at_utc FROM runtime_transfers ORDER BY id"
            )
        ]
    finally:
        connection.close()
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["runtime_transfer"] = {
        "history": history,
        "single_hardware_latency_benchmark": False,
        "definition": "Saved scores and latencies retain their source GPU/PyTorch provenance; "
        "remaining predictions use the new runtime. Classification pools both segments.",
    }
    payload["evaluation_code"]["files"]["scripts/resume_laya_other_pc.py"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    report.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", default="models/cordis_laya_v0.1.1/cordis_h2020/en_tr_direct"
    )
    parser.add_argument("--data-dir", default="data/processed/cordis_h2020")
    parser.add_argument(
        "--output-dir", default="results/metrics/cordis_laya_v0.1.1_validation_tuned"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if sys.platform != "win32" or sys.version_info[:2] != (3, 13):
        parser.error("Use Windows and Python 3.13 for this saved evaluation.")
    import torch
    from scripts import evaluate_all as evaluate

    if (
        torch.__version__ != "2.7.1+cu128"
        or not torch.cuda.is_available()
        or not torch.cuda.is_bf16_supported()
    ):
        parser.error("Install torch==2.7.1 from the cu128 wheel index and use a BF16-capable GPU.")
    output = evaluate.resolve_path(args.output_dir)
    if not (output / "prediction_cache.sqlite3").is_file():
        parser.error("Copy the saved prediction_cache.sqlite3 into the output directory first.")
    original_cache = evaluate.PredictionCache

    class TransferredCache(original_cache):
        def __init__(self, path: Path, key: str, identity: dict[str, Any]) -> None:
            transfer_identity(path, key, identity)
            super().__init__(path, key, identity)

    evaluate.PredictionCache = TransferredCache
    command = [
        "--models",
        "laya",
        "--split",
        "validation",
        "--tune-threshold",
        "--resume",
        "--device",
        "cuda",
        "--batch-size",
        "1",
        "--model-dir",
        args.model_dir,
        "--data-dir",
        args.data_dir,
        "--output-dir",
        args.output_dir,
    ]
    if args.overwrite:
        command.append("--overwrite")
    report_path = output / "evaluation_summary.json"
    previous_stamp = report_path.stat().st_mtime_ns if report_path.exists() else None
    try:
        return evaluate.main(command)
    finally:
        evaluate.PredictionCache = original_cache
        if report_path.exists() and report_path.stat().st_mtime_ns != previous_stamp:
            annotate_report(output)


if __name__ == "__main__":
    raise SystemExit(main())
