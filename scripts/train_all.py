"""Train independent, audited neural benchmark tracks; never prepare datasets here."""

from __future__ import annotations

import argparse
import json
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
    result.add_argument("--datasets", nargs="+", choices=("sib200", "multifin", "mn_ds"))
    result.add_argument("--device", choices=("auto", "cpu", "cuda"))
    result.add_argument("--epochs", type=int)
    result.add_argument("--batch-size", type=int)
    result.add_argument("--accumulation", type=int)
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
    supported = {label for record in train for label in record["labels"]}
    validation = [record for record in track["validation"] if set(record["labels"]) <= supported][
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
    labels = {label for record in train for label in record["labels"]}
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
    if args.heldout_topic and datasets != ["sib200"]:
        parser().error("--heldout-topic requires --datasets sib200")
    for value in (args.epochs, args.batch_size, args.accumulation, args.smoke_records):
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
                }
                for attribute, setting in (
                    ("epochs", "epochs"),
                    ("batch_size", "batch_size"),
                    ("accumulation", "gradient_accumulation_steps"),
                ):
                    if getattr(args, attribute) is not None:
                        overrides["training"][setting] = getattr(args, attribute)
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
                model.train(train, validation)
                # Verify artifact reload and a real inference before declaring success.
                model.save(destination)
                reloaded = MODEL_TYPES[family]({"device": config["device"]})
                reloaded.load(destination)
                reloaded.predict(validation[0]["text"])
                summaries.append({"run": key, "status": "success", "artifact": str(destination)})
                del model, reloaded
            except Exception as exc:
                summaries.append({"run": key, "status": "failed", "error": str(exc)})
            print(json.dumps(summaries[-1]), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    return int(any(run["status"] == "failed" for run in summaries))


if __name__ == "__main__":
    raise SystemExit(main())
