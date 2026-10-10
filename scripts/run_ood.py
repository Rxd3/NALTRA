"""Held-out-root near-OOD benchmark for the locally trainable members.

Protocol: every project touching the held-out EuroSciVoc root is removed from training and
validation, so neither its texts nor its labels are seen. In-distribution test projects
have no held-out label; OOD test projects carry only held-out-root labels; projects that
mix both are excluded as ambiguous. The OOD score is 1 - max label probability. Inputs pass
the shared release audit and cross-partition isolation check before any filtering or fitting;
AUROC carries a stratified 2,000-resample bootstrap 95% interval.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.train_all import MODEL_TYPES, resolve_path  # noqa: E402

from naltra.data.manifest import compute_file_sha256, get_environment_metadata  # noqa: E402
from naltra.data.validation import (  # noqa: E402
    SPLITS,
    _check_translated_record,
    check_disjoint_partitions,
    load_audit_records,
)
from naltra.evaluation.bootstrap import resample_counts  # noqa: E402
from naltra.evaluation.matrix import ood_metrics  # noqa: E402
from naltra.models.classical import ESTIMATOR_FILE  # noqa: E402
from naltra.pipeline.preprocessing import normalize_text  # noqa: E402
from naltra.utils.provenance import commit  # noqa: E402

Record = dict[str, Any]
# Keys and types of every summary.json this script writes; such an output may be replaced by a
# rerun. The provenance keys are not required, so outputs written before them still count.
RUN_KEYS = {
    "heldout_root": str,
    "protocol": str,
    "taxonomy_sha256": str,
    "inputs": dict,
    "counts": dict,
    "results": dict,
    "models": dict,
}
LANGUAGES = ("en", "tr")
# Bootstrap seeds of the in-distribution and the OOD score resamples.
SEEDS = (42, 43)


def root_of(label: str, parents: Mapping[str, str | None]) -> str:
    while parents[label] is not None:
        label = parents[label]
    return label


def partition(
    records: Sequence[Record], heldout: str, parents: Mapping[str, str | None]
) -> tuple[list[Record], list[Record], list[Record]]:
    """Split into (no held-out label, only held-out-root labels, mixed)."""
    clean, only, mixed = [], [], []
    for record in records:
        roots = {root_of(label, parents) for label in record["labels_direct"]}
        bucket = clean if heldout not in roots else only if roots == {heldout} else mixed
        bucket.append(record)
    return clean, only, mixed


def drop_heldout_projects(
    parts: Sequence[list[Record]], heldout: str, parents: Mapping[str, str | None]
) -> list[Record]:
    """Pool language variants; a project any variant ties to the held-out root leaves whole."""
    pooled = [r for part in parts for r in part]
    _, only, mixed = partition(pooled, heldout, parents)
    held = {r["project_id"] for r in only + mixed}
    return [r for r in pooled if r["project_id"] not in held]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-dir", default="data/processed/cordis_h2020")
    result.add_argument("--languages", nargs="+", choices=LANGUAGES, default=["en", "tr"])
    result.add_argument("--models", nargs="+", choices=("naive_bayes", "svm"), default=["svm"])
    result.add_argument("--heldout-root", default="humanities")
    result.add_argument("--output-dir", default="results/ood/humanities")
    result.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing --output-dir that holds no previous run_ood output; "
        "inside the repository only directories under results/",
    )
    result.add_argument("--taxonomy", default=str(REPO_ROOT / "taxonomy/taxonomy.json"))
    return result


def ood_scores(model: Any, records: list[Record]) -> np.ndarray:
    # Same normalisation as the models apply in train() and predict_batch().
    scores = model._probabilities([normalize_text(record["text"]) for record in records])
    return 1.0 - scores.max(axis=1)


def auroc_interval(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    resamples: int = 2000,
    seeds: tuple[int, int] = SEEDS,
) -> tuple[float, float]:
    """Stratified percentile bootstrap 95% interval; ID and OOD scores resample separately.

    AUROC is P(OOD score > ID score) plus half the ties, so each replicate is a weighted sum
    over the fixed pairwise win matrix.
    """
    wins = (id_scores[:, None] < ood_scores) + 0.5 * (id_scores[:, None] == ood_scores)
    id_counts = resample_counts(len(id_scores), resamples, seeds[0])
    ood_counts = resample_counts(len(ood_scores), resamples, seeds[1])
    aurocs = ((id_counts @ wins) * ood_counts).sum(axis=1) / wins.size
    low, high = np.percentile(aurocs, [2.5, 97.5])
    return float(low), float(high)


def audited_inputs(
    files: Mapping[tuple[str, str], Path], taxonomy: Path
) -> dict[tuple[str, str], list[Record]]:
    """Audit every file, then keep records, projects, pairs and content in one partition."""
    loaded = {key: load_audit_records(path, taxonomy) for key, path in files.items()}
    # One row per project and language, or a project could score as both ID and OOD.
    for (lang, split), part in loaded.items():
        counts = Counter(r["project_id"] for r in part)
        repeated = sorted(project for project, count in counts.items() if count > 1)
        if repeated:
            raise ValueError(f"{lang} {split} has several rows for project {repeated[0]}.")
    for split in SPLITS:
        projects = {
            lang: {r["project_id"] for r in part}
            for (lang, s), part in loaded.items()
            if s == split
        }
        if len({frozenset(ids) for ids in projects.values()}) > 1:
            raise ValueError(f"Languages cover different projects in {split}; not one release.")
    check_disjoint_partitions(
        {
            split: [r for (_, s), part in loaded.items() if s == split for r in part]
            for split in SPLITS
        }
    )
    # Translated siblings must keep the release invariants, labels included, or one
    # language's row could keep a held-out project in training.
    if {lang for lang, _ in loaded} == {"en", "tr"}:
        for split in SPLITS:
            sources = {r["project_id"]: r for r in loaded["en", split]}
            for r in loaded["tr", split]:
                try:
                    _check_translated_record(r, sources[r["project_id"]], strict=False)
                except ValueError as exc:
                    raise ValueError(f"{split} project {r['project_id']}: {exc}") from exc
    return loaded


def evaluate(
    family: str, taxonomy: Path, train: list[Record], validation: list[Record], tests: dict
) -> tuple[Any, dict[str, Any]]:
    model = MODEL_TYPES[family]({"target_field": "labels_direct"}, taxonomy_path=taxonomy)
    model.train(train, validation)
    result: dict[str, Any] = {"trained_labels": model.labels}
    for lang, (clean, only, _) in tests.items():
        id_scores, heldout_scores = ood_scores(model, clean), ood_scores(model, only)
        metrics = ood_metrics(id_scores, heldout_scores)
        low, high = auroc_interval(id_scores, heldout_scores)
        result[lang] = {
            "id_records": len(clean),
            "ood_records": len(only),
            **metrics,
            "auroc_ci_low": low,
            "auroc_ci_high": high,
        }
    print(json.dumps({family: {k: v for k, v in result.items() if k != "trained_labels"}}))
    return model, result


def model_identity(model: Any, directory: Path, revision: str | None) -> dict[str, Any]:
    model.save(directory)
    return {
        "config": model.config,
        "metadata": model.metadata,
        "taxonomy": model.taxonomy_identity,
        "commit": revision,
        "naltra_json_sha256": compute_file_sha256(directory / "naltra.json"),
        "estimator_sha256": compute_file_sha256(directory / ESTIMATOR_FILE),
    }


def family_result(result: Any) -> bool:
    """Whether ``result`` holds one family's trained labels and per-language scores."""
    if not (isinstance(result, dict) and isinstance(result.get("trained_labels"), list)):
        return False
    languages = result.keys() - {"trained_labels"}
    return (
        bool(languages)
        and languages <= set(LANGUAGES)
        and all(isinstance(result[language], dict) for language in languages)
    )


def previous_run(output: Path) -> bool:
    """Whether ``output`` holds a summary.json shaped like the ones this script writes."""
    try:
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        isinstance(summary, dict)
        and all(isinstance(summary.get(key), kind) for key, kind in RUN_KEYS.items())
        and bool(summary["results"])
        and summary["results"].keys() == summary["models"].keys()
        and all(isinstance(model, dict) for model in summary["models"].values())
        and all(family_result(result) for result in summary["results"].values())
    )


def output_refusal(output: Path, inputs: Sequence[Path], overwrite: bool) -> str | None:
    """Why ``output`` must not be written, or None: publishing deletes what it replaces."""
    for source in (path.resolve() for path in inputs):
        if output.is_relative_to(source) or source.is_relative_to(output):
            return f"--output-dir {output} overlaps the input {source}."
    root = REPO_ROOT.resolve()
    if root.is_relative_to(output):
        return "--output-dir must not be the repository or a directory holding it."
    if output.is_file():
        return f"--output-dir {output} is an existing file."
    # Inside the repository only results/ holds outputs that --overwrite may delete.
    if (
        overwrite
        and output.is_relative_to(root)
        and not output.parent.is_relative_to(root / "results")
    ):
        return "--overwrite replaces only directories under results/ or outside the repository."
    if output.exists() and not (overwrite or previous_run(output)):
        return f"{output} exists but holds no run_ood output; pass --overwrite to replace it."
    return None


def publish(staged: Path, output: Path, previous: Path) -> Path | None:
    """Swap the complete staged output in by renames, restoring the previous one on failure.

    ``previous`` must lie outside any temporary cleanup: if restoring fails too, the previous
    output stays there to be recovered. It is deleted only once the new output is installed;
    if that deletion fails, its path is returned.
    """
    if output.exists():
        output.rename(previous)
    try:
        staged.rename(output)
    except OSError:
        if previous.exists():
            previous.rename(output)
        raise
    shutil.rmtree(previous, ignore_errors=True)
    return previous if previous.exists() else None


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    taxonomy_path = resolve_path(args.taxonomy)
    # Models derive their taxonomy identity from the directory's taxonomy.json/label_map.json.
    if (
        taxonomy_path.name != "taxonomy.json"
        or not (taxonomy_path.parent / "label_map.json").is_file()
    ):
        parser().error("--taxonomy must be a taxonomy.json next to its label_map.json.")
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    parents = {item["id"]: item["parent"] for item in taxonomy["labels"]}
    if parents.get(args.heldout_root, "missing") is not None:
        parser().error(f"{args.heldout_root!r} is not a root of the taxonomy.")
    data, output = resolve_path(args.data_dir), resolve_path(args.output_dir).resolve()
    refusal = output_refusal(output, (data, taxonomy_path.parent), args.overwrite)
    if refusal:
        parser().error(refusal)
    files = {
        (lang, split): data / lang / f"{split}.jsonl" for lang in args.languages for split in SPLITS
    }
    try:
        loaded = audited_inputs(files, taxonomy_path)
        output.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), flush=True)
        return 1
    train, validation = (
        drop_heldout_projects(
            [loaded[lang, split] for lang in args.languages], args.heldout_root, parents
        )
        for split in ("train", "validation")
    )
    tests = {
        lang: partition(loaded[lang, "test"], args.heldout_root, parents) for lang in args.languages
    }
    revision = commit()
    summary: dict[str, Any] = {
        "heldout_root": args.heldout_root,
        "protocol": " ".join(__doc__.split("\n\n")[1].split()),
        "code_commit": revision,
        "environment": get_environment_metadata(),
        "bootstrap_seeds": {"id": SEEDS[0], "ood": SEEDS[1]},
        "taxonomy_sha256": compute_file_sha256(taxonomy_path),
        "inputs": {
            f"{lang}/{split}": compute_file_sha256(path) for (lang, split), path in files.items()
        },
        "counts": {
            "train": len(train),
            "validation": len(validation),
            **{
                f"{lang}_test_{kind}": len(part)
                for lang, parts in tests.items()
                for kind, part in zip(("id", "ood", "mixed_excluded"), parts, strict=True)
            },
        },
        "results": {},
        "models": {},
    }
    # Stage the whole output beside it (one filesystem, so the swap is two renames) and publish
    # only once every family is scored and saved: a failure leaves the previous output or none,
    # or, if it cannot be restored, keeps it at the reported sibling "<output>.previous-<suffix>".
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as tmp:
        staging = Path(tmp) / "new"
        previous = output.with_name(
            f"{output.name}.previous{Path(tmp).name.removeprefix(f'.{output.name}')}"
        )
        try:
            trained = {
                family: evaluate(family, taxonomy_path, train, validation, tests)
                for family in args.models
            }
            for family, (model, result) in trained.items():
                summary["results"][family] = result
                summary["models"][family] = model_identity(
                    model, staging / "models" / family, revision
                )
            (staging / "summary.json").write_text(
                json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            leftover = publish(staging, output, previous)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            failure = {"status": "failed", "error": str(exc)}
            if previous.exists():
                failure["previous_output"] = str(previous)
            print(json.dumps(failure), flush=True)
            return 1
    done = {"status": "success", "output": str(output)}
    if leftover is not None:
        done["previous_output"] = str(leftover)  # replaced, but it could not be deleted
    print(json.dumps(done), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
