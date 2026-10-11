"""Classify texts with saved artifacts through the shared inference pipeline.

Prints one JSON line per text: the selected labels (highest score first), the labels
closed under the taxonomy, the detected language and the OOD decision. One model keeps
its own thresholds and validation-fitted OOD threshold; several models are combined per
label by the ensemble voter. --benchmark-settings loads frozen thresholds and votes
from an evaluate_ensemble summary, after verifying the artifact and taxonomy hashes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
# Run as a script, Python only adds scripts/ to the path; sibling scripts need the root.
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.predict_all import artifact_hashes, load_model  # noqa: E402
from scripts.train_all import MODEL_TYPES, resolve_path  # noqa: E402

from naltra.data.manifest import compute_file_sha256  # noqa: E402
from naltra.pipeline import EnsembleConfig, PredictionPipeline  # noqa: E402
from naltra.pipeline.thresholds import is_probability  # noqa: E402
from naltra.schemas.prediction import PredictionResult  # noqa: E402
from naltra.utils.resources import resource_path  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--artifacts", required=True, help="Directory holding one artifact per model family"
    )
    result.add_argument("--models", nargs="+", choices=sorted(MODEL_TYPES), default=["svm"])
    result.add_argument(
        "--method",
        choices=("hard", "hard_k", "soft", "weighted_soft"),
        default="hard",
        help="Vote used for several models; hard_k/weighted_soft require --benchmark-settings",
    )
    result.add_argument("--benchmark-settings", help="Frozen evaluate_ensemble summary.json")
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--input", help="JSONL file with a 'text' field per record")
    return result


def summary(result: PredictionResult) -> dict[str, Any]:
    return {
        "text": result.text,
        "model": result.model,
        "labels": [asdict(item) for item in result.labels],
        "closed_labels": sorted({node for path in result.hierarchy_paths for node in path}),
        "language": asdict(result.language),
        "ood": {**asdict(result.ood), "method": result.metadata["ood_method"]},
    }


def is_text(value: Any) -> bool:
    """A non-blank string that UTF-8 can encode; a lone surrogate such as a JSON "\\ud800"
    escape cannot be, and would only fail later in language detection."""
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def input_texts(path: Path) -> list[str]:
    """Each non-blank line's text; a line that is not a JSON object with a non-empty string
    'text' is a usage error naming its line number, and so is a file without any text."""
    texts: list[str] = []
    with path.open("rb") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (ValueError, RecursionError):  # bad JSON, not UTF-8, or nested too deep
                record = None
            text = record.get("text") if isinstance(record, dict) else None
            if not is_text(text):
                parser().error(
                    f"--input line {number} must be a JSON object with a non-empty string 'text' "
                    "that UTF-8 can encode."
                )
            texts.append(text)
    if not texts:
        parser().error(f"--input {path} has no texts.")
    return texts


def benchmark_config(
    path: Path, models: dict, artifacts: Path, method: str
) -> EnsembleConfig | None:
    """Bind frozen decisions to the exact artifacts and label space that were evaluated."""
    settings = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("Benchmark settings must be a summary object.")
    if settings.get("taxonomy_sha256") != compute_file_sha256(
        resource_path("taxonomy/taxonomy.json")
    ):
        raise ValueError("Benchmark taxonomy differs from the current taxonomy.")
    thresholds = settings.get("thresholds")
    dumps = settings.get("dumps")
    if not isinstance(thresholds, dict) or not isinstance(dumps, dict):
        raise ValueError("Benchmark settings need thresholds and artifact provenance.")
    for family, model in models.items():
        if not is_probability(thresholds.get(family)):
            raise ValueError(f"Benchmark threshold is missing or invalid for {family}.")
        if settings.get("label_count") != len(model.labels):
            raise ValueError(f"Benchmark label universe differs for {family}.")
        actual = artifact_hashes(artifacts / family)
        origins = [v for k, v in dumps.items() if k.startswith(f"{family}/")]
        if not origins or any(
            not isinstance(origin, dict)
            or any(
                origin.get(key) != actual[key]
                for key in ("artifact_sha256", "artifact_files_sha256")
            )
            for origin in origins
        ):
            raise ValueError(f"Benchmark artifact hashes differ for {family}.")
    config = None
    if len(models) > 1:
        members = settings.get("members")
        if (
            not isinstance(members, list)
            or len(members) != len(models)
            or set(members) != set(models)
        ):
            raise ValueError("Benchmark voting members must exactly match --models.")
        if method == "hard_k":
            config = EnsembleConfig(
                method="hard", threshold=0.0, required_models=models, vote_count=settings.get("k")
            )
            if config.vote_count is None:
                raise ValueError("Benchmark settings have no tuned vote count.")
        elif method == "hard":
            config = EnsembleConfig(method="hard", required_models=models)
        else:
            key = "soft_threshold" if method == "soft" else "weighted_soft_threshold"
            if not is_probability(settings.get(key)):
                raise ValueError(f"Benchmark settings have no valid {key}.")
            weights = {}
            if method == "weighted_soft":
                values = settings.get("val_a_micro_f1")
                if not isinstance(values, dict) or any(
                    not is_probability(values.get(family)) for family in models
                ):
                    raise ValueError("Benchmark member weights must be finite scores in [0, 1].")
                weights = {family: values[family] for family in models}
                if sum(weights.values()) <= 0:
                    raise ValueError("Benchmark member weights must have a positive sum.")
            config = EnsembleConfig(
                method=method, threshold=settings[key], weights=weights, required_models=models
            )
    # Change decisions only after every identity and setting has validated.
    for family, model in models.items():
        model.config["multilabel"] = {"threshold": thresholds[family], "per_label": {}}
    return config


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    artifacts = resolve_path(args.artifacts)
    if len(set(args.models)) != len(args.models):
        parser().error("--models must contain distinct model families.")
    if args.method in {"hard_k", "weighted_soft"} and not args.benchmark_settings:
        parser().error(f"--method {args.method} requires --benchmark-settings.")
    missing = [family for family in args.models if not (artifacts / family).is_dir()]
    if missing:
        parser().error(f"No saved artifact for {missing} under {artifacts}.")
    if args.input is not None and not resolve_path(args.input).is_file():
        parser().error(f"No input file {resolve_path(args.input)}.")
    if args.text is not None and not is_text(args.text):
        parser().error("--text must be non-empty and encodable as UTF-8.")
    texts = [args.text] if args.text is not None else input_texts(resolve_path(args.input))
    models = {}
    for family in args.models:
        try:
            models[family] = load_model(family, artifacts, "auto")[0]
        except Exception as error:  # an incomplete or incompatible artifact
            parser().error(
                f"Cannot load the {family} artifact under {artifacts / family}: "
                f"{type(error).__name__}: {error}"
            )
    config = None
    provenance = None
    if args.benchmark_settings:
        settings_path = resolve_path(args.benchmark_settings)
        try:
            config = benchmark_config(settings_path, models, artifacts, args.method)
            provenance = {
                "benchmark_sha256": compute_file_sha256(settings_path),
                "method": args.method if len(models) > 1 else "single",
            }
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser().error(f"Cannot use benchmark settings: {error}")
    if len(models) == 1:
        pipeline = PredictionPipeline(model=next(iter(models.values())))
    else:
        config = config or EnsembleConfig(method=args.method, required_models=models)
        pipeline = PredictionPipeline(models=models, ensemble_config=config)
    for result in pipeline.predict_batch(texts):
        row = summary(result)
        if provenance is not None:
            row["threshold_source"] = provenance
        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
