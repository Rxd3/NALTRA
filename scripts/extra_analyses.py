"""Label coverage, zero-shot member pairs and error analysis behind the README results.

coverage  results/metrics/label_coverage.json: CORDIS codes the >= 50-project support filter
          drops without roll-up, and hierarchical F1 against the complete CORDIS closure.
pairwise  results/metrics/cordis_v1.1.0_sample/pairwise.json: zero-shot vs trained members on
          the 120-pair sample (paired project bootstrap, Holm-adjusted within each language).
errors    results/metrics/cordis_v1.1.0_ensemble/error_analysis.json: micro-F1 by EuroSciVoc
          root, training label frequency and gold labels per project, plus the top errors.

CPU only. Reads the committed archive (in memory), taxonomy and splits, and the local dumps of
scripts/predict_all.py, which must be the ones whose hashes the committed summaries recorded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# Run as a script, Python only adds scripts/ to the path; sibling scripts need the root.
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.evaluate_ensemble import Space, load_split, with_holm  # noqa: E402
from scripts.predict_all import resolve_path  # noqa: E402

from naltra.data.loader import load_jsonl  # noqa: E402
from naltra.data.manifest import (  # noqa: E402
    compute_file_sha256,
    get_environment_metadata,
    get_source_config,
)
from naltra.evaluation.bootstrap import SEED, paired_bootstrap  # noqa: E402
from naltra.evaluation.matrix import binarize, flat_metrics  # noqa: E402
from naltra.utils.provenance import commit, repo_path  # noqa: E402

ARCHIVE = REPO_ROOT / "data/raw/cordis_h2020/cordis-h2020projects-csv.zip"
TAXONOMY = REPO_ROOT / "taxonomy/taxonomy.json"
SPLITS = {
    split: REPO_ROOT / f"data/processed/cordis_h2020/en/{split}.jsonl"
    for split in ("train", "validation", "test")
}
ENSEMBLE = REPO_ROOT / "results/metrics/cordis_v1.1.0_ensemble"
SAMPLE = REPO_ROOT / "results/metrics/cordis_v1.1.0_sample"
OUTPUTS = {
    "coverage": REPO_ROOT / "results/metrics/label_coverage.json",
    "pairwise": SAMPLE / "pairwise.json",
    "errors": ENSEMBLE / "error_analysis.json",
}
FULL_TEST = ("en_test", "tr_test")
ZERO_SHOT = ("jev", "kev", "laya")
TRAINED = ("transformer", "naive_bayes", "bilstm", "svm", "hybrid_knn")
ERROR_SYSTEMS = ("svm", "hybrid_knn")
# EN-train projects per direct label: tail < 70 <= mid < 200 <= head.
FREQUENCY_EDGES = (70, 200)
FREQUENCY_BUCKETS = ("tail_lt70", "mid_70_199", "head_ge200")
TOP_ERRORS = 10
DECIMALS = 4


class Tree(NamedTuple):
    """The 586-node EuroSciVoc tree, keyed by CORDIS code path (e.g. /23/53/365)."""

    node_by_code: dict[str, str]
    parents: dict[str, str | None]
    names: dict[str, str]
    supported: frozenset[str]  # codes with >= 50 direct projects: the labels_direct universe


def load_tree(path: Path = TAXONOMY) -> Tree:
    labels = json.loads(path.read_text(encoding="utf-8"))["labels"]
    code = {label["id"]: label["euroscivoc_code"] for label in labels}
    for label in labels:
        parent = label["parent"]
        if parent is not None and not code[label["id"]].startswith(code[parent] + "/"):
            raise ValueError(f"{label['id']}: code is not below its parent's code path.")
    return Tree(
        node_by_code={value: key for key, value in code.items()},
        parents={label["id"]: label["parent"] for label in labels},
        names={label["id"]: label["name"] for label in labels},
        supported=frozenset(code[item["id"]] for item in labels if item["is_direct_supported"]),
    )


def parse_codes(text: str) -> dict[str, set[str]]:
    """Project id -> assigned codes, parsed like preprocessing.process_cordis_h2020."""
    codes: dict[str, set[str]] = defaultdict(set)
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=";")
    next(reader, None)
    for row in reader:
        if len(row) >= 5 and row[0].strip() and row[4].strip():
            codes[row[4].strip()].add(row[0].strip())
    return dict(codes)


def cordis_codes(archive: Path = ARCHIVE) -> dict[str, set[str]]:
    """Codes CORDIS assigns each project, read from the locked archive without extracting it."""
    with zipfile.ZipFile(archive) as bundle:
        names = [name for name in bundle.namelist() if Path(name).name == "euroSciVoc.csv"]
        if len(names) != 1:
            raise ValueError(f"Expected one euroSciVoc.csv in {repo_path(archive)}.")
        data = bundle.read(names[0])
    if hashlib.sha256(data).hexdigest() != get_source_config("cordis_h2020")["euroscivoc_sha256"]:
        raise ValueError("euroSciVoc.csv differs from the locked CORDIS release.")
    return parse_codes(data.decode("utf-8-sig"))


def code_prefixes(code: str) -> list[str]:
    """The code's ancestor paths, root first, then the code itself."""
    parts = code.strip("/").split("/")
    return ["/" + "/".join(parts[:depth]) for depth in range(1, len(parts) + 1)]


def implied_nodes(codes: Iterable[str], tree: Tree) -> set[str]:
    """The complete CORDIS closure: every tree node on the code path of an assigned code."""
    return {
        tree.node_by_code[prefix]
        for code in codes
        for prefix in code_prefixes(code)
        if prefix in tree.node_by_code
    }


def nearest_supported(code: str, tree: Tree) -> str | None:
    """The label a roll-up would give a dropped code: its nearest supported ancestor."""
    for prefix in reversed(code_prefixes(code)[:-1]):
        if prefix in tree.supported:
            return tree.node_by_code[prefix]
    return None


def coverage_counts(records: Iterable[dict], codes: Mapping[str, set[str]], tree: Tree) -> dict:
    """How far the stored labels of retained projects fall short of what CORDIS assigns."""
    counts: Counter[str] = Counter()
    for record in records:
        assigned = codes.get(record["project_id"], set())
        dropped = assigned - tree.supported
        missing = implied_nodes(assigned, tree) - set(record["labels"])
        roots = {node for node in missing if tree.parents[node] is None}
        rolled = {nearest_supported(code, tree) for code in dropped}
        counts["projects"] += 1
        counts["projects_losing_codes"] += bool(dropped)
        counts["codes_lost"] += len(dropped)
        counts["projects_rollup_would_add_label"] += bool(
            rolled - {None} - set(record["labels_direct"])
        )
        counts["closures_missing_nodes"] += bool(missing)
        counts["nodes_missing"] += len(missing)
        counts["projects_missing_root"] += bool(roots)
        counts["roots_missing"] += len(roots)
    return dict(counts)


def read_summary(directory: Path) -> dict:
    return json.loads((directory / "summary.json").read_text(encoding="utf-8"))


def load_checked(
    predictions: Path, directory: Path, summary: dict, families: Sequence[str], name: str
) -> dict[str, Any]:
    """A committed test set and its selections, only from the dumps the summary scored."""
    entry = summary["test_sets"][name]
    path = REPO_ROOT / entry["path"]
    if compute_file_sha256(path) != entry["sha256"]:
        raise ValueError(f"{entry['path']} differs from the test set {repo_path(directory)} used.")
    data = load_split(predictions, list(families), [(name, path)])
    for key, origin in data["dumps"].items():
        if origin["npz_sha256"] != summary["dumps"][key]["npz_sha256"]:
            raise ValueError(f"{key}: local dump differs from the one {repo_path(directory)} used.")
    scores = data["scores"]
    return {
        "rows": data["rows"],
        "labels": data["labels"],
        "truth": binarize([record["labels_direct"] for record in data["rows"]], data["labels"]),
        "selected": {f: scores[f] >= summary["thresholds"][f] for f in families},
    }


def hierarchical(gold: np.ndarray, closed: np.ndarray, which: str) -> dict[str, float]:
    metrics = flat_metrics(gold, closed)
    return {f"h{key}_{which}": metrics[key] for key in ("micro_precision", "micro_recall")} | {
        f"hmicro_f1_{which}": metrics["micro_f1"]
    }


def closure_effect(
    predictions: Path, summary: dict, codes: Mapping[str, set[str]], tree: Tree
) -> dict:
    """Hierarchical micro-F1 of each member against the stored and the complete gold closure.

    Needs every member's dumps, so a run without them fails instead of writing nulls.
    """
    families = summary["members"] + summary["baselines"]
    missing = [
        f"{f}/{s}"
        for f in families
        for s in FULL_TEST
        if not (predictions / f / f"{s}.npz").exists()
    ]
    if missing:
        raise FileNotFoundError(f"Prediction dumps missing: {', '.join(missing)}.")
    result = {}
    for name in FULL_TEST:
        data = load_checked(predictions, ENSEMBLE, summary, families, name)
        space = Space(data["labels"], TAXONOMY)
        stored = space.close(data["truth"])
        complete = binarize(
            [implied_nodes(codes.get(r["project_id"], set()), tree) for r in data["rows"]],
            space.nodes,
        )
        systems = {}
        for family in families:
            closed = space.close(data["selected"][family])
            systems[family] = hierarchical(stored, closed, "stored") | hierarchical(
                complete, closed, "complete"
            )
        result[name] = {"systems": systems, **closure_summary(systems)}
    return result


def closure_summary(systems: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    def ranking(which: str) -> list[str]:
        return sorted(systems, key=lambda family: -systems[family][f"hmicro_f1_{which}"])

    change = [abs(v["hmicro_f1_complete"] - v["hmicro_f1_stored"]) for v in systems.values()]
    return {
        "max_abs_hmicro_f1_change": max(change),
        "ranking_unchanged": ranking("stored") == ranking("complete"),
    }


def provenance(inputs: Iterable[Path]) -> dict[str, Any]:
    return {
        "code_commit": commit(),
        "environment": get_environment_metadata(),
        "inputs": {repo_path(path): compute_file_sha256(path) for path in inputs},
    }


def label_coverage(predictions: Path) -> dict[str, Any]:
    tree, codes = load_tree(), cordis_codes()
    per_split = {
        name: coverage_counts(load_jsonl(path), codes, tree) for name, path in SPLITS.items()
    }
    keys = next(iter(per_split.values()))
    per_split["all"] = {key: sum(counts[key] for counts in per_split.values()) for key in keys}
    summary = read_summary(ENSEMBLE)
    effect = closure_effect(predictions, summary, codes, tree)
    inputs = [ARCHIVE, TAXONOMY, *SPLITS.values(), ENSEMBLE / "summary.json"]
    return {
        **provenance(inputs),
        "definitions": {
            "codes_lost": "assigned codes below the 50-project support, dropped, not rolled up",
            "projects_rollup_would_add_label": "a dropped code's nearest supported ancestor "
            "is not in labels_direct",
            "nodes_missing": "tree nodes on assigned code paths that the stored closure lacks",
            "complete": "gold closure of every assigned code, as CORDIS implies it",
        },
        "counts": per_split,
        "hierarchical_micro_f1": effect,
        "thresholds": summary["thresholds"],
    }


def member_pairs(
    truth: np.ndarray, selected: Mapping[str, np.ndarray], groups: Sequence[str], resamples: int
) -> dict[str, dict]:
    """Each zero-shot member minus each trained member, Holm-adjusted across the family."""
    tests = {
        f"{a}-{b}": paired_bootstrap(
            truth, selected[a], truth, selected[b], resamples, groups=groups
        )
        for a in ZERO_SHOT
        for b in TRAINED
    }
    return with_holm(tests)


def pairwise(predictions: Path, resamples: int) -> dict[str, Any]:
    summary, families = read_summary(SAMPLE), ZERO_SHOT + TRAINED
    comparisons, resolution, projects = {}, {}, {}
    for name in summary["test_sets"]:
        data = load_checked(predictions, SAMPLE, summary, families, name)
        pairs = [record["pair_id"] for record in data["rows"]]
        tests = member_pairs(data["truth"], data["selected"], pairs, resamples)
        comparisons[name], projects[name] = tests, len(pairs)
        half = [(t["micro_f1"]["ci_high"] - t["micro_f1"]["ci_low"]) / 2 for t in tests.values()]
        resolution[name] = {"mean": float(np.mean(half)), "min": min(half), "max": max(half)}
    return {
        **provenance([SAMPLE / "summary.json"]),
        "resamples": resamples,
        "bootstrap_seed": SEED,
        "projects": projects,
        "thresholds": {family: summary["thresholds"][family] for family in families},
        "systems": {family: summary["benchmark"]["systems"][family] for family in families},
        "micro_f1_ci_half_width": resolution,
        "comparisons": comparisons,
    }


def root_of(node: str, parents: Mapping[str, str | None]) -> str:
    while (parent := parents[node]) is not None:
        node = parent
    return node


def micro_f1(truth: np.ndarray, chosen: np.ndarray) -> float:
    """2TP / (2TP + FP + FN) over every cell; sklearn would read one column as binary."""
    truth, chosen = truth.astype(bool), chosen.astype(bool)
    tp, errors = int((truth & chosen).sum()), int((truth ^ chosen).sum())
    return 2 * tp / (2 * tp + errors) if tp or errors else 0.0


def f1_by_group(
    truth: np.ndarray,
    systems: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    axis: int,
    unit: str,
) -> dict[str, dict]:
    """Micro-F1 of every system on the projects (axis 0) or label columns (axis 1) of a group.

    An empty group scores None.
    """
    result = {}
    for group, mask in masks.items():
        gold = np.compress(mask, truth, axis=axis)
        result[group] = {unit: int(mask.sum()), "gold_positives": int(gold.sum())} | {
            name: micro_f1(gold, np.compress(mask, chosen, axis=axis)) if mask.any() else None
            for name, chosen in systems.items()
        }
    return result


def frequency_masks(support: np.ndarray) -> dict[str, np.ndarray]:
    bucket = np.searchsorted(FREQUENCY_EDGES, support, side="right")
    return {name: bucket == index for index, name in enumerate(FREQUENCY_BUCKETS)}


def gold_count_masks(truth: np.ndarray) -> dict[str, np.ndarray]:
    count = truth.sum(axis=1)
    return {"1": count == 1, "2": count == 2, "3": count == 3, "4+": count >= 4}


def top_errors(
    truth: np.ndarray, chosen: np.ndarray, names: Sequence[str], top: int = TOP_ERRORS
) -> dict[str, dict[str, str]]:
    """Labels most often missed (false negatives) and most often wrongly predicted."""
    truth, chosen = truth.astype(bool), chosen.astype(bool)

    def ranked(errors: np.ndarray, totals: np.ndarray) -> dict[str, str]:
        order = np.argsort(-errors, kind="stable")[:top]
        return {names[i]: f"{errors[i]}/{totals[i]}" for i in order}

    return {
        "missed_of_gold": ranked((truth & ~chosen).sum(axis=0), truth.sum(axis=0)),
        "false_of_predicted": ranked((~truth & chosen).sum(axis=0), chosen.sum(axis=0)),
    }


def set_errors(data: dict, support: np.ndarray, tree: Tree) -> dict[str, Any]:
    truth, labels, selected = data["truth"], data["labels"], data["selected"]
    focus = {family: selected[family] for family in ERROR_SYSTEMS}
    roots = [root_of(label, tree.parents) for label in labels]
    root_masks = {tree.names[r]: np.array([x == r for x in roots]) for r in sorted(set(roots))}
    names = [tree.names[label] for label in labels]
    return {
        "label_frequency": f1_by_group(truth, selected, frequency_masks(support), 1, "labels"),
        "root": f1_by_group(truth, focus, root_masks, 1, "labels"),
        "gold_labels_per_project": f1_by_group(
            truth, focus, gold_count_masks(truth), 0, "projects"
        ),
        "top_errors": {family: top_errors(truth, focus[family], names) for family in focus},
    }


def error_analysis(predictions: Path) -> dict[str, Any]:
    summary, tree = read_summary(ENSEMBLE), load_tree()
    families = summary["members"] + summary["baselines"]
    train = [record["labels_direct"] for record in load_jsonl(SPLITS["train"])]
    sets = {}
    for name in FULL_TEST:
        data = load_checked(predictions, ENSEMBLE, summary, families, name)
        support = binarize(train, data["labels"]).sum(axis=0)  # in this set's column order
        sets[name] = set_errors(data, support, tree)
    return {
        **provenance([ENSEMBLE / "summary.json", SPLITS["train"], TAXONOMY]),
        "thresholds": summary["thresholds"],
        "label_frequency_support": "EN-train projects per direct label: tail < 70, mid 70-199, "
        "head >= 200; micro-F1 over that bucket's label columns",
        "sets": sets,
    }


def rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, DECIMALS)
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    result.add_argument("analyses", nargs="*", choices=sorted(OUTPUTS), help="default: all")
    result.add_argument("--predictions", default="results/predictions/cordis_v1.1.0")
    result.add_argument("--resamples", type=int, default=10000, help="Paired bootstrap resamples")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.resamples < 1:
        parser().error("--resamples must be positive.")
    predictions = resolve_path(args.predictions)
    builders: dict[str, Callable[[], dict]] = {
        "coverage": lambda: label_coverage(predictions),
        "pairwise": lambda: pairwise(predictions, args.resamples),
        "errors": lambda: error_analysis(predictions),
    }
    for name in dict.fromkeys(args.analyses or sorted(OUTPUTS)):
        try:
            payload = builders[name]()
        except (OSError, ValueError, KeyError) as exc:
            print(json.dumps({"status": "failed", "analysis": name, "error": str(exc)}), flush=True)
            return 1
        OUTPUTS[name].write_text(
            json.dumps(rounded(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(json.dumps({"status": "success", "output": repo_path(OUTPUTS[name])}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
