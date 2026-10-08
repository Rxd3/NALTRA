"""Evaluate saved neural artifacts on audited CORDIS partitions without training."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from naltra.data.manifest import compute_file_sha256, get_environment_metadata  # noqa: E402
from naltra.data.validation import validate_cordis_release  # noqa: E402
from naltra.evaluation.hierarchical import expand_with_ancestors  # noqa: E402
from naltra.evaluation.metrics import multilabel_metrics  # noqa: E402
from naltra.models.bilstm import BiLSTMModel  # noqa: E402
from naltra.models.neural import seed_everything  # noqa: E402
from naltra.models.transformer import TransformerModel  # noqa: E402
from naltra.pipeline.preprocessing import normalize_text  # noqa: E402
from naltra.pipeline.thresholds import apply_thresholds  # noqa: E402
from naltra.schemas.prediction import PredictionResult  # noqa: E402
from naltra.utils.config import load_yaml  # noqa: E402

MODEL_TYPES = {"bilstm": BiLSTMModel, "transformer": TransformerModel}


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def select_global_threshold(
    records: list[dict[str, Any]],
    predictions: list[PredictionResult],
    labels: list[str],
) -> dict[str, Any]:
    """Search a fixed grid on validation only; share one cutoff across both languages."""
    if not records or len(records) != len(predictions):
        raise ValueError("Threshold selection requires aligned, nonempty validation records.")
    if any(record.get("split") != "validation" for record in records):
        raise ValueError("Threshold selection is restricted to validation records.")
    probabilities = np.array([[p.label_scores[label] for label in labels] for p in predictions])
    outcomes = np.array([[label in r["labels_direct"] for label in labels] for r in records])
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid threshold selection probabilities.")
    positives = int(outcomes.sum())
    if not positives:
        raise ValueError("Threshold selection requires positive validation targets.")
    candidates = []
    for index in range(101):
        threshold = index / 100
        selected = probabilities >= threshold
        tp = int(np.count_nonzero(selected & outcomes))
        count = int(selected.sum())
        candidates.append(
            {
                "threshold": threshold,
                "micro_precision": tp / count if count else 0.0,
                "micro_recall": tp / positives,
                "micro_f1": 2 * tp / (count + positives),
            }
        )
    best = max(candidates, key=lambda candidate: (candidate["micro_f1"], candidate["threshold"]))
    return {
        "split": "validation",
        "objective": "pooled bilingual micro_f1",
        "method": "global grid 0.00 through 1.00 in steps of 0.01",
        "tie_break": "highest threshold",
        "records": len(records),
        "selected_threshold": best["threshold"],
        "candidates": candidates,
        "scores_selected_on_this_split": True,
    }


def frozen_thresholds(
    report: dict[str, Any], family: str, result: dict[str, Any], release: dict[str, Any]
) -> dict[str, Any]:
    """Reuse a full validation choice only with the same artifact and data release."""
    if (report.get("status"), report.get("split"), report.get("scope")) != (
        "success",
        "validation",
        "full",
    ):
        raise ValueError("Threshold source must be a successful full validation report.")
    previous = report["models"][family]
    if previous.get("threshold_selection", {}).get("split") != "validation":
        raise ValueError("Threshold source does not contain a validation-selected threshold.")
    if previous["artifact_sha256"] != result["artifact_sha256"]:
        raise ValueError("Threshold source model artifact hashes differ.")
    if report["label_count"] != len(release["label_universe"]):
        raise ValueError("Threshold source label universe differs.")
    for language in report["languages"]:
        for field in ("output_files", "taxonomy"):
            if (
                report["dataset_manifests"][language][field]
                != release["manifests"][language][field]
            ):
                raise ValueError("Threshold source data release differs.")
    thresholds = previous["thresholds"]
    apply_thresholds({}, thresholds["threshold"], thresholds["per_label"])
    if set(thresholds["per_label"]) - set(release["label_universe"]):
        raise ValueError("Threshold source contains unsupported labels.")
    return thresholds


def score_predictions(
    records: list[dict[str, Any]],
    predictions: list[PredictionResult],
    labels: list[str],
    parents: dict[str, str | None],
    bins: int,
) -> dict[str, Any]:
    """Use a fixed direct-label universe and every probability for calibration."""
    if not records or len(records) != len(predictions) or bins < 1:
        raise ValueError("Scoring needs nonempty, aligned records/predictions and positive bins.")
    universe = set(labels)
    if not labels or len(labels) != len(universe):
        raise ValueError("Evaluation labels must be nonempty and unique.")
    for record, prediction in zip(records, predictions, strict=True):
        if prediction.text != normalize_text(record["text"]):
            raise ValueError("Prediction order or source text differs from evaluation records.")
        if set(prediction.label_scores) != universe:
            raise ValueError("Probabilities must cover the exact direct-label universe.")
        if not {item.label for item in prediction.labels} <= universe:
            raise ValueError("Predicted labels fall outside the direct-label universe.")
    actual = [set(record["labels_direct"]) for record in records]
    predicted = [{item.label for item in prediction.labels} for prediction in predictions]
    direct = multilabel_metrics(actual, predicted, labels)
    direct["samples_jaccard"] = float(
        np.mean(
            [
                len(a & p) / len(a | p) if a | p else 1.0
                for a, p in zip(actual, predicted, strict=True)
            ]
        )
    )
    direct["subset_accuracy"] = float(
        np.mean([a == p for a, p in zip(actual, predicted, strict=True)])
    )
    direct["mean_true_labels"] = float(np.mean([len(a) for a in actual]))
    direct["mean_predicted_labels"] = float(np.mean([len(p) for p in predicted]))
    direct["empty_prediction_rate"] = float(np.mean([not p for p in predicted]))
    hierarchical = multilabel_metrics(
        [expand_with_ancestors(a, parents) for a in actual],
        [expand_with_ancestors(p, parents) for p in predicted],
        parents,
    )
    probabilities = np.array(
        [[p.label_scores[label] for label in labels] for p in predictions], dtype=np.float64
    )
    outcomes = np.array([[label in a for label in labels] for a in actual], dtype=np.float64)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid evaluation probabilities.")
    indices = np.minimum((probabilities.ravel() * bins).astype(int), bins - 1)
    count = np.bincount(indices, minlength=bins)
    confidence = np.bincount(indices, weights=probabilities.ravel(), minlength=bins)
    positive = np.bincount(indices, weights=outcomes.ravel(), minlength=bins)
    calibration = {
        "brier_score": float(np.mean((probabilities - outcomes) ** 2)),
        "ece": float(np.abs(positive - confidence).sum() / probabilities.size),
        "bins": bins,
        "definition": "binary labelwise calibration over every record and all direct labels",
        "reliability_bins": [
            {
                "lower": i / bins,
                "upper": (i + 1) / bins,
                "count": int(count[i]),
                "mean_probability": float(confidence[i] / count[i]) if count[i] else None,
                "positive_rate": float(positive[i] / count[i]) if count[i] else None,
            }
            for i in range(bins)
        ],
    }
    return {
        "records": len(records),
        "classification": direct,
        "hierarchical": hierarchical,
        "calibration": calibration,
        "ranking": {
            "micro_average_precision": float(
                average_precision_score(outcomes, probabilities, average="micro")
            ),
            "samples_average_precision": float(
                average_precision_score(outcomes, probabilities, average="samples")
            ),
        },
    }


def check_artifact(model: Any, release: dict[str, Any], base: Path) -> None:
    if model.config.get("target_field") != "labels_direct":
        raise ValueError("CORDIS evaluation requires a model trained on labels_direct.")
    if model.metadata.get("smoke") is not False:
        raise ValueError("Evaluation requires a full trained artifact, not a smoke model.")
    if model.metadata.get("dataset") != "cordis_h2020":
        raise ValueError("Artifact was not trained on CORDIS H2020.")
    if model.labels != release["label_universe"]:
        raise ValueError("Artifact does not cover the exact supported direct-label universe.")
    # Preserve the held-out contract: evaluate the same data release used in training.
    for language in model.metadata["training_languages"]:
        old = model.metadata["dataset_manifests"][language]
        current = json.loads((base / language / "manifest.json").read_text(encoding="utf-8"))
        for field in ("output_files", "taxonomy"):
            if old[field] != current[field]:
                raise ValueError(f"Artifact training release differs from current {language} data.")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model-dir", default="models/cordis_v0.4.0/cordis_h2020/en_tr_direct")
    result.add_argument("--data-dir", default="data/processed/cordis_h2020")
    result.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_TYPES), default=list(MODEL_TYPES)
    )
    result.add_argument("--languages", nargs="+", choices=("en", "tr"), default=["en", "tr"])
    result.add_argument("--split", choices=("validation", "test"), default="test")
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument(
        "--batch-size", type=int, help="Inference batch size; defaults to artifact."
    )
    result.add_argument(
        "--max-records", type=int, help="Per-language limit for a marked subset run."
    )
    result.add_argument("--output-dir", default="results/metrics/cordis_v0.4.0")
    result.add_argument("--overwrite", action="store_true")
    thresholds = result.add_mutually_exclusive_group()
    thresholds.add_argument(
        "--tune-threshold",
        action="store_true",
        help="Select each model's global threshold on full bilingual validation only.",
    )
    thresholds.add_argument(
        "--thresholds-file", help="Reuse thresholds from a full validation tuning summary."
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if any(value is not None and value < 1 for value in (args.batch_size, args.max_records)):
        parser().error("--batch-size and --max-records must be positive.")
    if len(set(args.models)) != len(args.models) or len(set(args.languages)) != len(args.languages):
        parser().error("Models and languages must be unique.")
    if args.tune_threshold and (
        args.split != "validation" or args.max_records or set(args.languages) != {"en", "tr"}
    ):
        parser().error("--tune-threshold requires full bilingual validation without --max-records.")
    output = resolve_path(args.output_dir) / "evaluation_summary.json"
    if output.exists() and not args.overwrite:
        parser().error("Evaluation output exists; use a new --output-dir or --overwrite.")
    base, model_dir = resolve_path(args.data_dir), resolve_path(args.model_dir)
    try:
        print("Auditing CORDIS data before loading models...", flush=True)
        release = validate_cordis_release(base, args.languages)
        bins = load_yaml(REPO_ROOT / "configs/evaluation.yaml")["calibration"]["bins"]
        threshold_report = (
            json.loads(resolve_path(args.thresholds_file).read_text(encoding="utf-8"))
            if args.thresholds_file
            else None
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Evaluation blocked: {exc}", flush=True)
        return 1
    summary: dict[str, Any] = {
        "status": "success",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "split": args.split,
        "scope": "subset" if args.max_records else "full",
        "records_per_language_limit": args.max_records,
        "languages": args.languages,
        "target_field": "labels_direct",
        "label_count": len(release["label_universe"]),
        "environment": get_environment_metadata(),
        "taxonomy": release["manifests"]["en"]["taxonomy"],
        "dataset_manifests": release["manifests"],
        "arguments": vars(args),
        "models": {},
        "evaluation_code": {
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
                ).strip()
            ),
            "files": {
                str(p.relative_to(REPO_ROOT)): compute_file_sha256(p)
                for p in [
                    Path(__file__),
                    REPO_ROOT / "src/naltra/evaluation/metrics.py",
                    REPO_ROOT / "src/naltra/evaluation/hierarchical.py",
                    REPO_ROOT / "src/naltra/models/neural.py",
                    REPO_ROOT / "src/naltra/models/bilstm/model.py",
                    REPO_ROOT / "src/naltra/models/transformer/model.py",
                    REPO_ROOT / "src/naltra/pipeline/preprocessing.py",
                    REPO_ROOT / "src/naltra/pipeline/thresholds.py",
                    REPO_ROOT / "configs/evaluation.yaml",
                ]
            },
        },
    }
    for family in args.models:
        model = None
        try:
            print(f"Loading saved {family}...", flush=True)
            model = MODEL_TYPES[family]({"device": args.device})
            model.load(model_dir / family)
            check_artifact(model, release, base)
            seed_everything(model.config["seed"])
            if args.batch_size:
                model.config["training"]["batch_size"] = args.batch_size
            result: dict[str, Any] = {
                "status": "success",
                "artifact": str(model_dir / family),
                "artifact_sha256": {
                    p.name: compute_file_sha256(p)
                    for p in sorted((model_dir / family).iterdir())
                    if p.is_file()
                },
                "device": str(model.device),
                "seed": model.config["seed"],
                "hardware": (
                    torch.cuda.get_device_name(model.device)
                    if model.device.type == "cuda"
                    else "CPU"
                ),
                "batch_size": model.config["training"]["batch_size"],
                "thresholds": model.config["multilabel"],
                "threshold_source": "saved_artifact",
                "slices": {},
            }
            if threshold_report is not None:
                result["artifact_thresholds"] = model.config["multilabel"]
                model.config["multilabel"] = frozen_thresholds(
                    threshold_report, family, result, release
                )
                result["thresholds"] = model.config["multilabel"]
                result["threshold_source"] = {
                    "validation_report": str(resolve_path(args.thresholds_file)),
                    "sha256": compute_file_sha256(resolve_path(args.thresholds_file)),
                }
            all_records, all_predictions = [], []
            for language in args.languages:
                records = release["corpora"][language][args.split]
                if args.max_records:
                    records = records[: args.max_records]
                predictions = []
                batch_size = model.config["training"]["batch_size"]
                start = perf_counter()
                for offset in range(0, len(records), batch_size):
                    batch = records[offset : offset + batch_size]
                    predictions.extend(model.predict_batch([r["text"] for r in batch]))
                    completed = min(offset + batch_size, len(records))
                    if completed == len(records) or completed % (batch_size * 100) == 0:
                        print(f"{family}/{language}: {completed}/{len(records)}", flush=True)
                elapsed = perf_counter() - start
                metrics = score_predictions(records, predictions, model.labels, model.parents, bins)
                metrics["prediction_seconds"] = elapsed
                result["slices"][language] = metrics
                print(
                    f"{family}/{language}: "
                    f"micro-F1={metrics['classification']['micro_f1']:.4f}, "
                    f"macro-F1={metrics['classification']['macro_f1']:.4f}",
                    flush=True,
                )
                all_records.extend(records)
                all_predictions.extend(predictions)
            result["slices"]["combined"] = score_predictions(
                all_records, all_predictions, model.labels, model.parents, bins
            )
            if args.tune_threshold:
                selection = select_global_threshold(all_records, all_predictions, model.labels)
                result["threshold_selection"] = selection
                result["artifact_thresholds"] = result["thresholds"]
                result["thresholds"] = {
                    "threshold": selection["selected_threshold"],
                    "per_label": {},
                }
                result["threshold_source"] = "validation_search"
                result["baseline_slices"] = result["slices"]
                result["slices"] = {}
                adjusted = [
                    replace(
                        p, labels=apply_thresholds(p.label_scores, selection["selected_threshold"])
                    )
                    for p in all_predictions
                ]
                offset = 0
                for language in args.languages:
                    count = result["baseline_slices"][language]["records"]
                    result["slices"][language] = score_predictions(
                        all_records[offset : offset + count],
                        adjusted[offset : offset + count],
                        model.labels,
                        model.parents,
                        bins,
                    )
                    result["slices"][language]["prediction_seconds"] = result["baseline_slices"][
                        language
                    ]["prediction_seconds"]
                    offset += count
                result["slices"]["combined"] = score_predictions(
                    all_records, adjusted, model.labels, model.parents, bins
                )
                print(
                    f"{family}: validation-selected threshold="
                    f"{selection['selected_threshold']:.2f}",
                    flush=True,
                )
            summary["models"][family] = result
            print(
                f"{family}: combined micro-F1="
                f"{result['slices']['combined']['classification']['micro_f1']:.4f}, "
                f"macro-F1={result['slices']['combined']['classification']['macro_f1']:.4f}",
                flush=True,
            )
        except Exception as exc:
            summary["status"] = "failed"
            summary["models"][family] = {"status": "failed", "error": str(exc)}
            print(f"{family} evaluation failed: {exc}", flush=True)
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Evaluation {summary['status']}. Results: {output}", flush=True)
    return int(summary["status"] != "success")


if __name__ == "__main__":
    raise SystemExit(main())
