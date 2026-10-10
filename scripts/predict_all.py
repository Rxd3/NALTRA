"""Score evaluation sets with trained artifacts and save full label-score matrices.

Every downstream stage (thresholds, ensemble voting, metrics, calibration) reads these
dumps offline, so each model scores each set exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# Run as a script, Python only adds scripts/ to the path; sibling scripts need the root.
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.train_all import MODEL_TYPES, resolve_path  # noqa: E402

from naltra.data.loader import load_jsonl  # noqa: E402
from naltra.data.manifest import compute_file_sha256  # noqa: E402
from naltra.models.jev import JevModel  # noqa: E402
from naltra.models.kev import KevModel  # noqa: E402
from naltra.models.laya import LayaModel  # noqa: E402

# Inference services are built from their YAML config; they have no trained artifact.
SERVICE_TYPES = {"kev": KevModel, "jev": JevModel}
# Pretrained zero-shot members load a prepared artifact (scripts/prepare_laya.py) but are
# never trained by train_all.
PRETRAINED_TYPES = {"laya": LayaModel}

CORDIS = "data/processed/cordis_h2020"
EVAL_SETS = {
    f"{language}_{split}": f"{CORDIS}/{language}/{split}.jsonl"
    for language in ("en", "tr")
    for split in ("validation", "test")
} | {
    f"cs_{strategy}_{split}": f"{CORDIS}/code_switch/{strategy}_mix/balanced/{split}.jsonl"
    for strategy in ("chunk", "sentence")
    for split in ("validation", "test")
}


def repeated_names(*groups: Sequence[tuple[str, Path]]) -> list[str]:
    """Set names given more than once; each names its dump file, so names must be unique."""
    names = [name for group in groups for name, _ in group]
    return sorted({name for name in names if names.count(name) > 1})


def parse_set(value: str) -> tuple[str, Path]:
    """Accept a registered set name or an explicit name=path pair."""
    name, separator, path = value.partition("=")
    if separator:
        # Set names become file names and "family/set" provenance keys.
        if not name or any(mark in name for mark in ("/", "\\", "..")):
            raise argparse.ArgumentTypeError(f"Set name {name!r} must be a plain file stem.")
        return name, resolve_path(path)
    if value not in EVAL_SETS:
        raise argparse.ArgumentTypeError(f"Unknown set {value!r}; use one of {sorted(EVAL_SETS)}")
    return value, resolve_path(EVAL_SETS[value])


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--artifacts", required=True, help="Directory holding one artifact per model family"
    )
    result.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_TYPES | SERVICE_TYPES | PRETRAINED_TYPES),
        default=["svm"],
    )
    result.add_argument(
        "--sets", nargs="+", type=parse_set, default=[parse_set("en_test"), parse_set("tr_test")]
    )
    result.add_argument("--output-dir", default="results/predictions")
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--limit", type=int, help="Score only the first N records (pilot runs)")
    result.add_argument("--overwrite", action="store_true")
    return result


def score_set(model, texts: list[str], batch_size: int) -> tuple[np.ndarray, float, dict]:
    scores = np.empty((len(texts), len(model.labels)), dtype=np.float32)
    latency, metadata = 0.0, {}
    for start in range(0, len(texts), batch_size):
        results = model.predict_batch(texts[start : start + batch_size])
        metadata = metadata or results[0].metadata
        for offset, result in enumerate(results):
            scores[start + offset] = [result.label_scores[label] for label in model.labels]
            latency += result.latency_ms
        print(f"  {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return scores, latency / max(1, len(texts)), metadata


def load_model(family: str, artifacts: Path, device: str) -> tuple[Any, Path | None]:
    if family in SERVICE_TYPES:
        return SERVICE_TYPES[family](), None
    model = (PRETRAINED_TYPES | MODEL_TYPES)[family]({"device": device})
    model.load(artifacts / family)
    return model, artifacts / family


def artifact_hashes(artifact: Path | None) -> dict[str, Any]:
    """Hash every artifact file, so neural weights and tokenizers are covered too."""
    if artifact is None:
        return {"artifact_sha256": None, "artifact_files_sha256": None, "artifact_files": None}
    files = {
        file.relative_to(artifact).as_posix(): compute_file_sha256(file)
        for file in artifact.rglob("*")
        if file.is_file()
    }
    files = dict(sorted(files.items()))
    return {
        "artifact_sha256": files["naltra.json"],
        "artifact_files_sha256": hashlib.sha256(
            json.dumps(list(files.items())).encode()
        ).hexdigest(),
        "artifact_files": files,
    }


def dump(model, family: str, hashes: dict, name: str, path: Path, args, output: Path) -> dict:
    target = output / family / f"{name}.npz"
    if target.exists() and not args.overwrite:
        raise FileExistsError(f"{target} exists; pass --overwrite to replace it.")
    records = load_jsonl(path)[: args.limit]
    if not records:
        raise ValueError(f"{path} has no records to score.")
    # Only the classified text is scored; TR records also carry their English source.
    texts = [record["text"] for record in records]
    scores, latency, metadata = score_set(model, texts, args.batch_size)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        ids=np.array([record["id"] for record in records]),
        labels=np.array(model.labels),
        scores=scores,
    )
    sidecar = {
        "model": family,
        "eval_set": name,
        "input": str(path),
        "input_sha256": compute_file_sha256(path),
        "records": len(records),
        "limit": args.limit,
        **hashes,
        "config_sha256": compute_file_sha256(REPO_ROOT / "configs/models" / f"{family}.yaml"),
        "threshold": model.config["multilabel"]["threshold"],
        "per_label_thresholds": model.config["multilabel"]["per_label"],
        "mean_latency_ms": latency,
        "metadata": {k: v for k, v in metadata.items() if k != "supported_labels"},
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    target.with_suffix(".json").write_text(
        json.dumps(sidecar, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return {"run": f"{family}/{name}", "status": "success", "output": str(target)}


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if repeated := repeated_names(args.sets):
        parser().error(f"Each set name must be unique; repeated: {', '.join(repeated)}")
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser().error("--batch-size and --limit must be positive integers.")
    artifacts, output = resolve_path(args.artifacts), resolve_path(args.output_dir)
    summaries = []
    for family in args.models:
        try:
            model, artifact = load_model(family, artifacts, args.device)
            hashes = artifact_hashes(artifact)
        except Exception as exc:
            for name, _ in args.sets:
                summaries.append({"run": f"{family}/{name}", "status": "failed", "error": str(exc)})
                print(json.dumps(summaries[-1]), flush=True)
            continue
        for name, path in args.sets:
            print(f"Scoring {family} on {name} ({path})", flush=True)
            try:
                summaries.append(dump(model, family, hashes, name, path, args, output))
            except Exception as exc:
                summaries.append({"run": f"{family}/{name}", "status": "failed", "error": str(exc)})
            print(json.dumps(summaries[-1]), flush=True)
    return int(any(run["status"] == "failed" for run in summaries))


if __name__ == "__main__":
    raise SystemExit(main())
