"""Measure warmed-up single-text latency, batched throughput and model size per family.

Latency is end to end through predict()/predict_batch(), so it includes tokenisation,
vectorisation and, for Kev, HTTP.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# The inference stack, and the variables that change what native code does.
RUNTIME_PACKAGES = (
    "torch",
    "transformers",
    "tokenizers",
    "huggingface-hub",
    "numpy",
    "scipy",
    "scikit-learn",
)
NATIVE_SETTINGS = ("CUDA_VISIBLE_DEVICES", "PYTORCH_NVML_BASED_CUDA_CHECK", "OMP_NUM_THREADS")
# Run as a script, Python only adds scripts/ to the path; sibling scripts need the root.
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.predict_all import (  # noqa: E402
    MODEL_TYPES,
    SERVICE_TYPES,
    artifact_hashes,
    load_model,
    parse_set,
    resolve_path,
)

from naltra.data.loader import load_jsonl  # noqa: E402
from naltra.data.manifest import compute_file_sha256, get_environment_metadata  # noqa: E402
from naltra.evaluation.latency import latency_profile  # noqa: E402
from naltra.models.classical import ClassicalModel  # noqa: E402
from naltra.models.external import ExternalServiceModel  # noqa: E402
from naltra.utils.provenance import commit, repo_path  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--artifacts", nargs="+", required=True, help="Run directories; first match per family"
    )
    result.add_argument("--models", nargs="+", choices=sorted(MODEL_TYPES | SERVICE_TYPES))
    result.add_argument("--set", type=parse_set, default=parse_set("en_test"))
    result.add_argument("--samples", type=int, default=200, help="Texts timed one at a time")
    result.add_argument("--warmup", type=int, default=10)
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--output-dir", default="results/benchmarks")
    return result


def artifact_dir(family: str, directories: list[str]) -> Path:
    """The first run directory that holds this family (members can come from different runs)."""
    runs = [resolve_path(directory) for directory in directories]
    return next((run for run in runs if (run / family).is_dir()), runs[0])


def parameter_count(model: Any) -> int | None:
    """Trainable parameters (neural) or stored weight entries (classical/kNN index)."""
    network = getattr(model, "network", None)
    if network is not None:
        return int(sum(p.numel() for p in network.parameters()))
    weights = getattr(model, "weights", None)
    if weights:
        return int(sum(getattr(array, "nnz", None) or np.size(array) for array in weights.values()))
    return None


def artifact_bytes(artifact: Path | None) -> int | None:
    if artifact is None:
        return None
    return sum(path.stat().st_size for path in artifact.rglob("*") if path.is_file())


def hardware(device: str) -> dict[str, Any]:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
    }
    try:
        import torch

        if device != "cpu" and torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def runtime() -> dict[str, Any]:
    """Installed inference dependencies and native execution settings of this timing run."""
    packages = {}
    for package in RUNTIME_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    info: dict[str, Any] = {
        "packages": packages,
        "environment": {name: os.environ.get(name) for name in NATIVE_SETTINGS},
    }
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["torch_cuda"] = torch.version.cuda
        info["torch_threads"] = torch.get_num_threads()
    except ImportError:
        pass
    return info


def resolved_device(model: Any) -> str | None:
    """Where inference actually ran, never just the configured preference."""
    if isinstance(model, ExternalServiceModel):
        return "service"
    if isinstance(model, ClassicalModel):
        # Scoring is NumPy/SciPy; only a created dense encoder can run elsewhere.
        encoder_device = getattr(getattr(model, "_encoder", None), "device", None)
        return "cpu" if encoder_device is None else f"cpu+encoder:{encoder_device}"
    device = getattr(model, "device", None)
    return None if device is None else str(device)


def benchmark(model: Any, artifact: Path | None, texts: list[str], args) -> dict[str, Any]:
    single = latency_profile(lambda text: model.predict(text), texts, warmup=args.warmup)
    start = perf_counter()
    for offset in range(0, len(texts), args.batch_size):
        model.predict_batch(texts[offset : offset + args.batch_size])
    elapsed = perf_counter() - start
    hashes = artifact_hashes(artifact)
    return {
        "single": single,
        "batched": {
            "batch_size": args.batch_size,
            "docs": len(texts),
            "docs_per_second": len(texts) / elapsed if elapsed else float("inf"),
        },
        "device": resolved_device(model),
        "artifact_sha256": hashes["artifact_sha256"],
        "artifact_files_sha256": hashes["artifact_files_sha256"],
        "artifact_bytes": artifact_bytes(artifact),
        "parameters": parameter_count(model),
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "| model | p50 ms | p95 ms | p99 ms | docs/s (batched) | size MB | parameters |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, row in summary["models"].items():
        size = row["artifact_bytes"]
        lines.append(
            f"| {name} | {row['single']['p50_ms']:.2f} | {row['single']['p95_ms']:.2f} | "
            f"{row['single']['p99_ms']:.2f} | {row['batched']['docs_per_second']:.1f} | "
            f"{'-' if size is None else f'{size / 1e6:.1f}'} | {row['parameters'] or '-'} |"
        )
    packages = summary["runtime"]["packages"].items()
    versions = ", ".join(f"{name} {version}" for name, version in packages)
    return "\n".join(lines) + f"\n\nRuntime: {versions}\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.samples < 1 or args.warmup < 0 or args.batch_size < 1:
        parser().error("--samples and --batch-size must be positive, --warmup nonnegative.")
    name, path = args.set
    texts = [record["text"] for record in load_jsonl(path)[: args.samples]]
    summary: dict[str, Any] = {
        "eval_set": name,
        "eval_set_path": repo_path(path),
        "eval_set_sha256": compute_file_sha256(path),
        "samples": len(texts),
        "warmup": args.warmup,
        "batch_size": args.batch_size,
        "requested_device": args.device,
        "code_commit": commit(),
        "environment": get_environment_metadata(),
        "hardware": hardware(args.device),
        "runtime": runtime(),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "models": {},
        "failures": {},
    }
    for family in args.models or ["svm"]:
        try:
            model, artifact = load_model(family, artifact_dir(family, args.artifacts), args.device)
            summary["models"][family] = benchmark(model, artifact, texts, args)
        except Exception as exc:
            summary["failures"][family] = str(exc)
        print(json.dumps({family: summary["models"].get(family) or summary["failures"][family]}))
    output = resolve_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "benchmark_summary.md").write_text(markdown(summary), encoding="utf-8")
    return int(bool(summary["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
