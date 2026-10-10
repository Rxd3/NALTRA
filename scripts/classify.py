"""Classify texts with saved artifacts through the shared inference pipeline.

Prints one JSON line per text: the selected labels (highest score first), the labels
closed under the taxonomy, the detected language and the OOD decision. One model keeps
its own thresholds and validation-fitted OOD threshold; several models are combined per
label by the ensemble voter. Ensemble votes use a fixed 0.5 soft threshold (not the
benchmark's tuned one) and report no OOD decision.
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

from scripts.predict_all import load_model  # noqa: E402
from scripts.train_all import MODEL_TYPES, resolve_path  # noqa: E402

from naltra.pipeline import EnsembleConfig, PredictionPipeline  # noqa: E402
from naltra.schemas.prediction import PredictionResult  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--artifacts", required=True, help="Directory holding one artifact per model family"
    )
    result.add_argument("--models", nargs="+", choices=sorted(MODEL_TYPES), default=["svm"])
    result.add_argument(
        "--method", choices=("hard", "soft"), default="hard", help="Vote used for several models"
    )
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    artifacts = resolve_path(args.artifacts)
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
    if len(models) == 1:
        pipeline = PredictionPipeline(model=next(iter(models.values())))
    else:
        config = EnsembleConfig(method=args.method, required_models=models)
        pipeline = PredictionPipeline(models=models, ensemble_config=config)
    for result in pipeline.predict_batch(texts):
        print(json.dumps(summary(result)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
