"""Train independent, audited classical and neural benchmark tracks; never prepare datasets."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from naltra.data.manifest import compute_file_sha256  # noqa: E402
from naltra.models.bilstm import BiLSTMModel  # noqa: E402
from naltra.models.hybrid_knn import HybridKNNModel  # noqa: E402
from naltra.models.naive_bayes import NaiveBayesModel  # noqa: E402
from naltra.models.neural import merge_config, seed_everything  # noqa: E402
from naltra.models.svm import SVMModel  # noqa: E402
from naltra.models.transformer import TransformerModel  # noqa: E402
from naltra.utils.config import load_yaml  # noqa: E402

MODEL_TYPES = {
    "naive_bayes": NaiveBayesModel,
    "svm": SVMModel,
    "hybrid_knn": HybridKNNModel,
    "bilstm": BiLSTMModel,
    "transformer": TransformerModel,
}


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/training.yaml")
    result.add_argument("--models", nargs="+", help="naive_bayes svm hybrid_knn bilstm transformer")
    result.add_argument("--datasets", nargs="+", choices=("cordis_h2020",))
    result.add_argument(
        "--languages",
        nargs="+",
        choices=("en", "tr"),
        help="CORDIS training languages (default: en tr)",
    )
    result.add_argument(
        "--prebuilt-release",
        action="store_true",
        help="Audit a distributed CORDIS release record-by-record (no raw sources/split map)",
    )
    result.add_argument("--device", choices=("auto", "cpu", "cuda"))
    result.add_argument("--epochs", type=int)
    result.add_argument("--batch-size", type=int)
    result.add_argument("--accumulation", type=int)
    result.add_argument("--learning-rate", type=float)
    result.add_argument("--max-length", type=int)
    result.add_argument("--loss-weighting", choices=("none", "sqrt_inverse_frequency"))
    result.add_argument("--max-positive-weight", type=float)
    result.add_argument(
        "--refit-head-from",
        help="Refit classifiers from the model family folders in this directory.",
    )
    result.add_argument("--feature-batch-size", type=int, default=8)
    result.add_argument("--output-dir")
    result.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Bounded smoke training; tiny random Transformer. Offline except hybrid_knn, "
            "which loads its e5 encoder from the Hugging Face cache (downloading it if absent)"
        ),
    )
    result.add_argument("--smoke-records", type=int)
    result.add_argument(
        "--overwrite", action="store_true", help="Replace a previously produced artifact"
    )
    return result


def prepare_track(dataset: str, spec: dict[str, Any]) -> dict[str, Any]:
    from naltra.data import validation

    directory = resolve_path(spec["path"])
    languages = spec["languages"]
    prebuilt = spec.get("release") == "prebuilt"
    audit = (
        validation.validate_prebuilt_cordis_release
        if prebuilt
        else validation.validate_cordis_release
    )
    release = audit(directory, languages)
    target_field = spec.get("target_field", "labels_direct")
    if target_field != "labels_direct":
        raise ValueError("The CORDIS baseline must train direct labels.")
    track = "_".join(languages) + "_direct"
    return {
        "train": [r for language in languages for r in release["corpora"][language]["train"]],
        "validation": [
            r for language in languages for r in release["corpora"][language]["validation"]
        ],
        "track": track,
        "target_field": target_field,
        "provenance": {
            "dataset": dataset,
            "track": track,
            "training_languages": languages,
            "target_field": target_field,
            "label_universe": release["label_universe"],
            "dataset_manifests": release["manifests"],
            "manifest_sha256_by_language": {
                language: compute_file_sha256(directory / language / "manifest.json")
                for language in languages
                if (directory / language / "manifest.json").exists()
            },
            "split_sha256": release["split_sha256"],
            "release_audit": release.get("audit", "full"),
            "translation_provenance": release.get("translation_provenance"),
            "translation_issues": release.get("translation_issues"),
        },
    }


def _keep_labels(records: list[dict[str, Any]], field: str, keep: set[str]) -> list[dict[str, Any]]:
    trimmed = [{**r, field: [label for label in r[field] if label in keep]} for r in records]
    return [record for record in trimmed if record[field]]


def smoke_records(
    track: dict[str, Any], limit: int, min_positives: int = 1
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = track["train"][:limit]
    field = track.get("target_field", "labels")
    if min_positives > 1:
        # Calibrated members need enough positives and negatives for every kept label.
        while True:
            counts: dict[str, int] = {}
            for record in train:
                for label in record[field]:
                    counts[label] = counts.get(label, 0) + 1
            keep = {
                label
                for label, count in counts.items()
                if min_positives <= count <= len(train) - min_positives
            }
            trimmed = _keep_labels(train, field, keep)
            if trimmed == train:
                break
            train = trimmed
        if len(keep) < 2:
            raise ValueError("Smoke subset leaves fewer than two trainable labels.")
    supported = {label for record in train for label in record[field]}
    candidates = (
        _keep_labels(track["validation"], field, supported)
        if min_positives > 1
        else [record for record in track["validation"] if set(record[field]) <= supported]
    )
    validation = candidates[:limit]
    if not validation:
        raise ValueError(
            "Smoke subset has no compatible validation records; increase --smoke-records."
        )
    return train, validation


def tiny_transformer(config: dict[str, Any], train: list[dict[str, Any]]) -> TransformerModel:
    """Actual local Transformer, randomly initialized; never a benchmark checkpoint."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        PreTrainedTokenizerFast,
        XLMRobertaConfig,
        XLMRobertaForSequenceClassification,
    )

    seed_everything(config["seed"])
    words = sorted({word for record in train for word in record["text"].split()})[:1000]
    vocabulary = {
        word: index for index, word in enumerate(["<pad>", "<unk>", "<s>", "</s>", *words])
    }
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
    )
    labels = {label for record in train for label in record[config.get("target_field", "labels")]}
    network = XLMRobertaForSequenceClassification(
        XLMRobertaConfig(
            vocab_size=len(vocabulary),
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            num_labels=len(labels),
            pad_token_id=0,
            bos_token_id=2,
            eos_token_id=3,
            max_position_embeddings=66,
            problem_type="multi_label_classification",
        )
    )
    config = merge_config(
        config,
        {
            "architecture": {"pretrained_name": "tiny-offline-smoke", "max_length": 32},
            "training": {"gradient_checkpointing": False},
        },
    )
    return TransformerModel(config, tokenizer=fast, network=network)


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    configuration = load_yaml(resolve_path(args.config))
    selected_models = args.models or configuration["models"]
    datasets = args.datasets or configuration["datasets"]
    if args.refit_head_from and (args.smoke or datasets != ["cordis_h2020"] or not args.output_dir):
        parser().error(
            "Head refitting requires CORDIS, no --smoke, and an explicit new --output-dir."
        )
    if args.learning_rate is not None and (
        not math.isfinite(args.learning_rate) or args.learning_rate <= 0
    ):
        parser().error("--learning-rate must be finite and positive.")
    if args.max_positive_weight is not None and (
        not math.isfinite(args.max_positive_weight) or args.max_positive_weight < 1
    ):
        parser().error("--max-positive-weight must be finite and at least 1.")
    if args.refit_head_from and args.accumulation not in (None, 1):
        parser().error("Cached head refitting requires --accumulation 1.")
    if args.languages:
        if datasets != ["cordis_h2020"] or len(set(args.languages)) != len(args.languages):
            parser().error("--languages requires unique languages and --datasets cordis_h2020.")
        configuration["tracks"]["cordis_h2020"]["languages"] = args.languages
    if args.prebuilt_release:
        if datasets != ["cordis_h2020"]:
            parser().error("--prebuilt-release requires --datasets cordis_h2020.")
        configuration["tracks"]["cordis_h2020"]["release"] = "prebuilt"
    for value in (
        args.epochs,
        args.batch_size,
        args.accumulation,
        args.smoke_records,
        args.max_length,
        args.feature_batch_size,
    ):
        if value is not None and value < 1:
            parser().error("Training counts must be positive integers.")
    unknown = set(selected_models) - set(MODEL_TYPES)
    if unknown:
        parser().error(
            f"Unsupported local training models: {sorted(unknown)}. "
            "Kev and Jev are inference services, not locally trained models."
        )
    if len(set(selected_models)) != len(selected_models) or len(set(datasets)) != len(datasets):
        parser().error("Model and dataset selections must be unique.")
    output = resolve_path(args.output_dir or configuration["output_dir"])
    tracks: dict[str, Any] = {}
    failures: dict[str, str] = {}
    # Audit all selected tracks before any model is initialized or weights downloaded.
    for dataset in datasets:
        try:
            tracks[dataset] = prepare_track(dataset, configuration["tracks"][dataset])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures[dataset] = str(exc)
    summaries = []
    for dataset in datasets:
        for family in selected_models:
            model = reloaded = None
            key = f"{dataset}/{family}"
            try:
                if dataset in failures:
                    raise ValueError(failures[dataset])
                track = tracks[dataset]
                config = load_yaml(REPO_ROOT / "configs/models" / f"{family}.yaml")
                if not config["enabled"]:
                    summaries.append({"run": key, "status": "disabled"})
                    print(json.dumps(summaries[-1]), flush=True)
                    continue
                overrides: dict[str, Any] = {
                    "device": args.device or configuration["device"],
                    "training": {},
                    "target_field": track.get("target_field", "labels"),
                }
                for attribute, setting in (
                    ("epochs", "epochs"),
                    ("batch_size", "batch_size"),
                    ("accumulation", "gradient_accumulation_steps"),
                    ("learning_rate", "learning_rate"),
                    ("loss_weighting", "loss_weighting"),
                    ("max_positive_weight", "max_positive_weight"),
                ):
                    if getattr(args, attribute) is not None:
                        overrides["training"][setting] = getattr(args, attribute)
                if args.max_length:
                    overrides["architecture"] = {"max_length": args.max_length}
                if args.smoke:
                    overrides["training"].setdefault("epochs", 1)
                    overrides = merge_config(overrides, {"architecture": {"max_length": 32}})
                    if family == "bilstm":
                        overrides = merge_config(
                            overrides, {"architecture": {"embedding_dim": 32, "hidden_dim": 32}}
                        )
                config = merge_config(config, overrides)
                train, validation = (
                    smoke_records(
                        track,
                        args.smoke_records or configuration["smoke_records"],
                        config.get("classifier", {}).get("calibration_cv", 1),
                    )
                    if args.smoke
                    else (track["train"], track["validation"])
                )
                destination = (
                    output / dataset / (track["track"] + ("_smoke" if args.smoke else "")) / family
                )
                if destination.exists() and not args.overwrite:
                    raise FileExistsError(
                        f"Artifact directory exists: {destination}; "
                        "use a new --output-dir or --overwrite."
                    )
                print(
                    f"Training {key}: {len(train)} train, {len(validation)} validation, "
                    f"device={config['device']}, smoke={args.smoke}",
                    flush=True,
                )
                model = (
                    tiny_transformer(config, train)
                    if args.smoke and family == "transformer"
                    else MODEL_TYPES[family](config)
                )
                if args.refit_head_from:
                    initial = resolve_path(args.refit_head_from) / family
                    if destination.resolve() == initial.resolve():
                        raise ValueError(
                            "Head refitting must preserve the initial artifact directory."
                        )
                    model.load(initial)
                    if model.metadata.get("smoke") is not False:
                        raise ValueError("Head refitting requires a full trained artifact.")
                    if model.config.get("target_field") != track["target_field"]:
                        raise ValueError("Initial artifact uses different targets.")
                    for language in track["provenance"]["training_languages"]:
                        old = model.metadata["dataset_manifests"][language]
                        current = track["provenance"]["dataset_manifests"][language]
                        if any(
                            old[field] != current[field] for field in ("output_files", "taxonomy")
                        ):
                            raise ValueError("Initial artifact uses a different data release.")
                    head_settings = {
                        "epochs": args.epochs or 20,
                        "batch_size": args.batch_size or 256,
                        "learning_rate": args.learning_rate or 0.001,
                        "gradient_accumulation_steps": args.accumulation or 1,
                        "patience": 3,
                        "mixed_precision": False,
                        "feature_batch_size": args.feature_batch_size,
                        "loss_weighting": args.loss_weighting or "none",
                        "max_positive_weight": (
                            args.max_positive_weight
                            if args.max_positive_weight is not None
                            else 20.0
                        ),
                    }
                    head_overrides = {
                        "training": head_settings,
                        "inference": {"batch_size": args.feature_batch_size},
                    }
                    if args.max_length:
                        head_overrides["architecture"] = {"max_length": args.max_length}
                    model.config = merge_config(model.config, head_overrides)
                    model.metadata["initial_artifact"] = {
                        "path": str(initial),
                        "sha256": {
                            p.name: compute_file_sha256(p)
                            for p in sorted(initial.iterdir())
                            if p.is_file()
                        },
                    }
                provenance_paths = [
                    Path(__file__),
                    REPO_ROOT / "src/naltra/models/neural.py",
                    REPO_ROOT / "src/naltra/models/classical.py",
                    REPO_ROOT / "src/naltra/models" / family / "model.py",
                    REPO_ROOT / "configs/models" / f"{family}.yaml",
                    resolve_path(args.config),
                ]
                model.metadata.update(
                    {
                        **track["provenance"],
                        "smoke": args.smoke,
                        "code_sha256": {
                            str(path.relative_to(REPO_ROOT)): compute_file_sha256(path)
                            for path in provenance_paths
                            if path.is_relative_to(REPO_ROOT)
                        },
                        "training_config": configuration,
                    }
                )
                if args.refit_head_from:
                    model.refit_head(train, validation)
                else:
                    model.train(train, validation)
                # Verify artifact reload and a real inference before declaring success.
                model.save(destination)
                reloaded = MODEL_TYPES[family]({"device": config["device"]})
                reloaded.load(destination)
                reloaded.predict(validation[0]["text"])
                summaries.append({"run": key, "status": "success", "artifact": str(destination)})
            except Exception as exc:
                summaries.append({"run": key, "status": "failed", "error": str(exc)})
            finally:
                del model, reloaded
                gc.collect()
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print(json.dumps(summaries[-1]), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    return int(any(run["status"] == "failed" for run in summaries))


if __name__ == "__main__":
    raise SystemExit(main())
