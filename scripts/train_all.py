"""Train independent, audited neural benchmark tracks; never prepare datasets here."""

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

from naltra.data.loader import load_jsonl  # noqa: E402
from naltra.data.manifest import (  # noqa: E402
    compute_file_sha256,
    validate_manifest,
)
from naltra.data.preprocessing import compute_content_fingerprint  # noqa: E402
from naltra.models.bilstm import BiLSTMModel  # noqa: E402
from naltra.models.neural import merge_config, seed_everything  # noqa: E402
from naltra.models.transformer import TransformerModel  # noqa: E402
from naltra.utils.config import load_yaml  # noqa: E402

MODEL_TYPES = {"bilstm": BiLSTMModel, "transformer": TransformerModel}


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/training.yaml")
    result.add_argument("--models", nargs="+", help="bilstm transformer (default: both)")
    result.add_argument(
        "--datasets", nargs="+", choices=("cordis_h2020", "sib200", "multifin", "mn_ds")
    )
    result.add_argument(
        "--languages",
        nargs="+",
        choices=("en", "tr"),
        help="CORDIS training languages (default: en tr)",
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
        help="Bounded, offline smoke training; tiny random Transformer",
    )
    result.add_argument("--smoke-records", type=int)
    result.add_argument("--heldout-topic", help="Use one existing SIB-200 near-OOD ID fold")
    result.add_argument(
        "--overwrite", action="store_true", help="Replace a previously produced artifact"
    )
    return result


def prepare_track(dataset: str, spec: dict[str, Any], heldout_topic: str | None) -> dict[str, Any]:
    directory = resolve_path(spec["path"])
    if dataset == "cordis_h2020":
        from naltra.data.validation import validate_cordis_release

        if heldout_topic:
            raise ValueError(
                "CORDIS held-out-domain training requires a separately defined protocol."
            )
        languages = spec["languages"]
        release = validate_cordis_release(directory, languages)
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
                },
            },
        }
    expected = ["train.jsonl", "validation.jsonl", "test.jsonl"]
    benchmark = spec["benchmark_name"]
    track = spec["name"]
    train_file, validation_file = "train.jsonl", "validation.jsonl"
    if heldout_topic:
        if dataset != "sib200":
            raise ValueError("--heldout-topic requires only the sib200 dataset.")
        # Never let an arbitrary path escape the known fold directory.
        canonical = {
            label["id"]
            for label in json.loads(
                (REPO_ROOT / "taxonomy/taxonomy.json").read_text(encoding="utf-8")
            )["labels"]
        }
        if heldout_topic not in canonical:
            raise ValueError("Unknown held-out topic.")
        directory = REPO_ROOT / "data/ood/near/sib200" / heldout_topic
        expected = [
            "train_id.jsonl",
            "validation_id.jsonl",
            "test_id.jsonl",
            "validation_ood.jsonl",
            "test_ood.jsonl",
        ]
        benchmark = f"near_ood_sib200_{heldout_topic}"
        track = f"heldout_{heldout_topic}"
        train_file, validation_file = "train_id.jsonl", "validation_id.jsonl"
    manifest = validate_manifest(directory, benchmark, expected, REPO_ROOT / "taxonomy")
    train, validation = load_jsonl(directory / train_file), load_jsonl(directory / validation_file)
    # The audit checks test-file integrity, but no test records enter this training path.
    if heldout_topic and any(heldout_topic in record["labels"] for record in train + validation):
        raise ValueError("Held-out topic appears in ID training/validation.")
    train_ids = {record["id"] for record in train}
    train_fingerprints = {compute_content_fingerprint(record["text"]) for record in train}
    if train_ids & {record["id"] for record in validation} or train_fingerprints & {
        compute_content_fingerprint(record["text"]) for record in validation
    }:
        raise ValueError("Training/validation contamination detected.")
    return {
        "train": train,
        "validation": validation,
        "track": track,
        "manifest": manifest,
        "provenance": {
            "dataset": dataset,
            "track": track,
            "manifest_sha256": compute_file_sha256(directory / "manifest.json"),
            "dataset_manifest": manifest,
            "train_sha256": compute_file_sha256(directory / train_file),
            "validation_sha256": compute_file_sha256(directory / validation_file),
        },
    }


def smoke_records(
    track: dict[str, Any], limit: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = track["train"][:limit]
    field = track.get("target_field", "labels")
    supported = {label for record in train for label in record[field]}
    validation = [record for record in track["validation"] if set(record[field]) <= supported][
        :limit
    ]
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
    if args.heldout_topic and datasets != ["sib200"]:
        parser().error("--heldout-topic requires --datasets sib200")
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
            "Classical models are not implemented; Jev/Laya are inference services."
        )
    if len(set(selected_models)) != len(selected_models) or len(set(datasets)) != len(datasets):
        parser().error("Model and dataset selections must be unique.")
    output = resolve_path(args.output_dir or configuration["output_dir"])
    tracks: dict[str, Any] = {}
    failures: dict[str, str] = {}
    # Audit all selected tracks before any model is initialized or weights downloaded.
    for dataset in datasets:
        try:
            tracks[dataset] = prepare_track(
                dataset, configuration["tracks"][dataset], args.heldout_topic
            )
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
                    smoke_records(track, args.smoke_records or configuration["smoke_records"])
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
