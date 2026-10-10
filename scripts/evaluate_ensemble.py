"""Offline benchmark: tune thresholds on validation, vote, and score every test set.

Reads the score matrices written by scripts/predict_all.py. Validation is split by pair_id:
val-A tunes each member's threshold, val-B tunes the ensemble settings, and test sets are
only scored.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# Run as a script, Python only adds scripts/ to the path; sibling scripts need the root.
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.predict_all import parse_set, repeated_names, resolve_path  # noqa: E402

from naltra.data.loader import load_jsonl  # noqa: E402
from naltra.data.manifest import compute_file_sha256, get_environment_metadata  # noqa: E402
from naltra.data.validation import check_disjoint_partitions  # noqa: E402
from naltra.evaluation.bootstrap import SEED, holm, paired_bootstrap  # noqa: E402
from naltra.evaluation.matrix import (  # noqa: E402
    binarize,
    calibration_metrics,
    close_upward,
    flat_metrics,
    hierarchy_violation_rate,
    label_distribution,
    ranking_metrics,
)
from naltra.evaluation.offline import (  # noqa: E402
    THRESHOLD_GRID,
    hard_vote,
    split_halves,
    strict_majority,
    tune_k,
    tune_threshold,
    weighted_soft_vote,
)
from naltra.utils.config import load_yaml  # noqa: E402
from naltra.utils.provenance import commit, repo_path  # noqa: E402

EPSILON = 1e-9
ENSEMBLES = ("hard_majority", "hard_k", "soft", "weighted_soft")
IDENTITY = ("artifact_sha256", "artifact_files_sha256", "config_sha256")
BENCHMARK = REPO_ROOT / "configs/benchmark.yaml"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--predictions", required=True, help="Directory of <family>/<set>.npz")
    result.add_argument("--members", nargs="+", required=True, help="Voting model families")
    result.add_argument("--baselines", nargs="*", default=[], help="Scored, never voting")
    result.add_argument(
        "--tune-sets", nargs="+", type=parse_set, default=[parse_set("en_validation")]
    )
    result.add_argument("--test-sets", nargs="+", type=parse_set, default=[parse_set("en_test")])
    result.add_argument("--output-dir", default="results/metrics/benchmark")
    result.add_argument(
        "--bootstrap", type=int, default=10000, help="Paired bootstrap resamples (0 disables)"
    )
    result.add_argument("--taxonomy", default=str(REPO_ROOT / "taxonomy/taxonomy.json"))
    return result


class Space:
    """Direct label columns, closed node columns and gold matrices for one label space."""

    def __init__(self, labels: list[str], taxonomy_path: Path) -> None:
        taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
        self.parents = {item["id"]: item["parent"] for item in taxonomy["labels"]}
        self.labels = labels
        nodes = set()
        for label in labels:
            node: str | None = label
            while node is not None:
                nodes.add(node)
                node = self.parents[node]
        self.nodes = sorted(nodes)

    def close(self, direct: np.ndarray) -> np.ndarray:
        return close_upward(direct, self.labels, self.nodes, self.parents)


def check_dump(where: str, labels: np.ndarray, scores: np.ndarray, rows: int) -> list[str]:
    names = labels.tolist()
    if labels.ndim != 1 or not names or not all(isinstance(n, str) and n for n in names):
        raise ValueError(f"{where}: labels must be a non-empty list of strings.")
    if len(set(names)) != len(names):
        raise ValueError(f"{where}: labels must be unique.")
    if scores.shape != (rows, len(names)):
        raise ValueError(f"{where}: scores shape {scores.shape} is not ({rows}, {len(names)}).")
    if not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError(f"{where}: scores must be finite probabilities in [0, 1].")
    return names


def load_scores(
    predictions: Path, family: str, name: str, ids: list[str], digest: str
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    """Load one dump whose sidecar proves it scored exactly this gold file with this family."""
    where, target = f"{family}/{name}", predictions / family / f"{name}.npz"
    sidecar = json.loads((predictions / family / f"{name}.json").read_text(encoding="utf-8"))
    if not isinstance(sidecar, dict):
        raise ValueError(f"{where}: sidecar is not a JSON object.")
    expected = {"model": family, "eval_set": name, "records": len(ids), "input_sha256": digest}
    stale = [key for key, value in expected.items() if sidecar.get(key) != value]
    if stale:
        raise ValueError(f"{where}: sidecar {stale} do not match the gold set; re-run predict_all.")
    with np.load(target) as dump:
        if dump["ids"].tolist() != ids:
            raise ValueError(f"{where}: dump rows do not match the gold set order.")
        scores = dump["scores"].astype(float)
        labels = check_dump(where, dump["labels"], scores, len(ids))
    origin = {"npz_sha256": compute_file_sha256(target), "input_sha256": digest}
    kept = {k: v for k, v in sidecar.items() if k.startswith("artifact") or k in IDENTITY}
    return labels, scores, origin | kept


def check_family_identity(dumps: dict[str, dict[str, Any]]) -> None:
    """Refuse a family whose dumps ("family/set" -> provenance) came from different models."""
    seen: dict[str, list] = {}
    for key, origin in dumps.items():
        family = key.rpartition("/")[0]
        identity = [origin.get(field) for field in IDENTITY]
        if seen.setdefault(family, identity) != identity:
            raise ValueError(f"{family}: dumps come from different artifacts/configs ({key}).")


def load_split(predictions: Path, families: list[str], sets: list) -> dict[str, Any]:
    """Concatenate sets; every family must share the gold row order and label columns."""
    rows = [record for _, path in sets for record in load_jsonl(path)]
    ids = {name: [r["id"] for r in load_jsonl(path)] for name, path in sets}
    digests = {name: compute_file_sha256(path) for name, path in sets}
    scores, labels, dumps = {}, None, {}
    for family in families:
        parts = []
        for name, _ in sets:
            family_labels, matrix, dumps[f"{family}/{name}"] = load_scores(
                predictions, family, name, ids[name], digests[name]
            )
            if labels not in (None, family_labels):
                raise ValueError(f"{family}/{name}: label columns differ from other members.")
            labels = family_labels
            parts.append(matrix)
        scores[family] = np.concatenate(parts)
    return {"rows": rows, "scores": scores, "labels": labels, "dumps": dumps}


def leakage(tune_sets: list, test_sets: list) -> str | None:
    """Name the shared test material a tuning set would leak, or None when they are disjoint."""
    tuning = [record for _, path in tune_sets for record in load_jsonl(path)]
    testing = [record for _, path in test_sets for record in load_jsonl(path)]
    if any(record.get("split") == "test" for record in tuning):
        return "a tuning set holds test-split records."
    try:
        check_disjoint_partitions({"tuning": tuning, "test": testing})
    except ValueError as exc:
        return str(exc)
    return None


def system_metrics(space: Space, truth: np.ndarray, selected: np.ndarray, scores=None) -> dict:
    gold_closed = space.close(truth)
    closed = space.close(selected)
    result = flat_metrics(truth, selected)
    result |= {f"h{k}": v for k, v in flat_metrics(gold_closed, closed).items() if "micro" in k}
    if scores is not None:
        result |= ranking_metrics(truth, scores) | calibration_metrics(truth, scores)
    return result


def tune(space: Space, data: dict, members: list[str], baselines: list[str]) -> dict:
    truth = binarize([r["labels_direct"] for r in data["rows"]], space.labels)
    half_b = split_halves([r["pair_id"] for r in data["rows"]])
    a, b = ~half_b, half_b
    thresholds, weights = {}, {}
    for family in members + baselines:
        thresholds[family], weights[family] = tune_threshold(truth[a], data["scores"][family][a])
    stack = np.stack([data["scores"][f][b] for f in members])
    selections = np.stack([s >= thresholds[f] for s, f in zip(stack, members, strict=True)])
    closed = np.stack([space.close(s) for s in selections])
    k, _ = tune_k(truth[b], hard_vote(selections), len(members))
    k_closed, _ = tune_k(space.close(truth[b]), hard_vote(closed), len(members))
    member_weights = np.array([weights[f] for f in members])
    soft_t, _ = tune_threshold(truth[b], stack.mean(axis=0))
    weighted_t, _ = tune_threshold(truth[b], weighted_soft_vote(stack, member_weights))
    return {
        "thresholds": thresholds,
        "val_a_micro_f1": weights,
        "k": k,
        "k_closed": k_closed,
        "soft_threshold": soft_t,
        "weighted_soft_threshold": weighted_t,
        "val_a_records": int(a.sum()),
        "val_b_records": int(b.sum()),
    }


def evaluate_set(space: Space, data: dict, members: list[str], baselines: list[str], p: dict):
    truth = binarize([r["labels_direct"] for r in data["rows"]], space.labels)
    scores = data["scores"]
    selected = {f: scores[f] >= p["thresholds"][f] for f in members + baselines}
    results = {f: system_metrics(space, truth, selected[f], scores[f]) for f in selected}
    count = len(members)
    fraction = hard_vote(np.stack([selected[f] for f in members]))
    chosen = dict(selected)
    chosen["hard_majority"] = fraction >= strict_majority(count) - EPSILON
    chosen["hard_k"] = fraction >= p["k"] / count - EPSILON
    results["hard_majority"] = system_metrics(space, truth, chosen["hard_majority"])
    results["hard_k"] = system_metrics(space, truth, chosen["hard_k"])
    closed = hard_vote(np.stack([space.close(selected[f]) for f in members]))
    closed_selected = closed >= p["k_closed"] / count - EPSILON
    # Closed-space votes add ancestors, so only hierarchical metrics are comparable.
    closed_metrics = flat_metrics(space.close(truth), closed_selected)
    results["hard_closed"] = {f"h{k}": v for k, v in closed_metrics.items() if "micro" in k}
    results["hard_closed"]["hierarchy_violation_rate"] = hierarchy_violation_rate(
        closed_selected, space.nodes, space.parents
    )
    stack = np.stack([scores[f] for f in members])
    soft = stack.mean(axis=0)
    weighted = weighted_soft_vote(stack, np.array([p["val_a_micro_f1"][f] for f in members]))
    chosen["soft"] = soft >= p["soft_threshold"]
    chosen["weighted_soft"] = weighted >= p["weighted_soft_threshold"]
    results["soft"] = system_metrics(space, truth, chosen["soft"], soft)
    results["weighted_soft"] = system_metrics(space, truth, chosen["weighted_soft"], weighted)
    leave_one_out = {}
    for dropped in members:
        kept = [f for f in members if f != dropped]
        votes = hard_vote(np.stack([selected[f] for f in kept]))
        majority = votes >= strict_majority(len(kept)) - EPSILON
        leave_one_out[dropped] = flat_metrics(truth, majority)["micro_f1"]
    pairs = [record["pair_id"] for record in data["rows"]]
    return results, leave_one_out, {"truth": truth, "pairs": pairs, "systems": chosen}


def with_holm(tests: dict[str, dict]) -> dict[str, dict]:
    """Add Holm-adjusted p-values across one family of comparisons, per metric."""
    for metric in ("micro_f1", "macro_f1"):
        adjusted = holm({name: result[metric]["p_value"] for name, result in tests.items()})
        for name, result in tests.items():
            result[metric]["p_holm"] = adjusted[name]
    return tests


def significance(chosen: dict, members: list[str], settings: dict, resamples: int) -> dict:
    """Ensembles vs the best member (chosen on val-A), and each system across conditions.

    Cross-condition comparisons pair projects by pair_id against the first test set, so
    EN vs TR / code-switch / noisy differences resample the same projects on both sides.
    Every comparison resamples pair_ids, so variants of one project count once.
    """
    best = max(members, key=lambda family: settings["val_a_micro_f1"][family])
    versus = {}
    for name, data in chosen.items():
        truth, systems, pairs = data["truth"], data["systems"], data["pairs"]
        versus[name] = with_holm(
            {
                system: paired_bootstrap(
                    truth, systems[system], truth, systems[best], resamples, groups=pairs
                )
                for system in ENSEMBLES
            }
        )
    reference, *others = list(chosen)
    base = chosen[reference]
    index = {pair: row for row, pair in enumerate(base["pairs"])}
    comparable = [
        name
        for name in others
        if len(index) == len(base["pairs"]) == len(chosen[name]["pairs"])
        and set(chosen[name]["pairs"]) == set(index)
    ]
    cross = {}
    for system in base["systems"]:
        tests = {}
        for name in comparable:
            data = chosen[name]
            order = [index[pair] for pair in data["pairs"]]
            tests[f"{reference}->{name}"] = paired_bootstrap(
                base["truth"][order],
                base["systems"][system][order],
                data["truth"],
                data["systems"][system],
                resamples,
                groups=data["pairs"],
            )
        cross[system] = with_holm(tests)
    return {
        "resamples": resamples,
        "best_member": best,
        "reference_set": reference,
        "vs_best_member": versus,
        "cross_condition": cross,
    }


def benchmark_systems(systems: list[str], produced: set[str]) -> dict[str, Any]:
    """Benchmark metrics plus the role, architecture and model of every scored system."""
    config = load_yaml(BENCHMARK)
    primary, secondary = config.get("primary_metric"), config.get("secondary_metrics")
    for key, names in (("primary_metric", [primary]), ("secondary_metrics", secondary)):
        if not (
            isinstance(names, list) and all(isinstance(n, str) and n in produced for n in names)
        ):
            raise ValueError(
                f"{BENCHMARK}: {key} must name metrics the summary produces: {sorted(produced)}"
            )
    entries = config.get("systems") or {}
    missing = [name for name in systems if name not in entries]
    if missing:
        raise ValueError(
            f"No benchmark entry for {missing} in {BENCHMARK}; add role, architecture and model."
        )
    fields = ("role", "architecture", "model")
    for name in systems:
        entry = entries[name]
        if not (
            isinstance(entry, dict)
            and entry.get("role") in ("baseline", "new")
            and all(isinstance(entry.get(field), str) and entry[field] for field in fields)
        ):
            raise ValueError(f"{BENCHMARK}: {name} needs role baseline|new, architecture, model.")
    return {
        "primary_metric": primary,
        "secondary_metrics": secondary,
        "systems": {name: {field: entries[name][field] for field in fields} for name in systems},
    }


def provenance(sets: list) -> dict[str, dict[str, str]]:
    return {
        name: {"path": repo_path(path), "sha256": compute_file_sha256(path)} for name, path in sets
    }


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def benchmark_markdown(summary: dict) -> list[str]:
    """Per test set, systems ranked by the primary metric, then the ablations; label imbalance."""
    benchmark = summary["benchmark"]
    primary = benchmark["primary_metric"]
    columns = [primary, *benchmark["secondary_metrics"]]
    header = "| rank | system | role | architecture | model | " + " | ".join(columns) + " |"
    intro = (
        f"Ranked by {primary} (ties share a rank); "
        "ablations and systems without it are unranked."
    )
    if "significance" in summary:
        intro += " Paired-difference intervals between systems and conditions: see Significance."
    lines = ["## Benchmark", "", intro, ""]
    for name, systems in summary["metrics"].items():
        scores = [values[primary] for values in systems.values() if primary in values]
        rows = sorted(
            systems.items(), key=lambda row: (primary not in row[1], -row[1].get(primary, 0.0))
        )
        # Like the results page, leave-one-out ablations are listed but never ranked.
        rows += [
            (f"leave_one_out(-{dropped})", {"micro_f1": f1})
            for dropped, f1 in summary["leave_one_out"][name].items()
        ]
        lines += [f"### {name}", "", header, "|" + "---|" * (5 + len(columns))]
        for system, values in rows:
            entry = benchmark["systems"][system.partition("(")[0]]
            # Competition ranking: one plus the systems strictly ahead, so ties share a rank.
            ranked = system in systems and primary in values
            cells = [str(1 + sum(s > values[primary] for s in scores)) if ranked else "-", system]
            cells += [entry[field] for field in ("role", "architecture", "model")]
            cells += [_cell(values.get(column)) for column in columns]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    distribution = benchmark["label_distribution"]
    lines += [
        "### Label distribution",
        "",
        "all_negative_accuracy is the per-label accuracy of predicting no label at all.",
        "",
        "| statistic | " + " | ".join(distribution) + " |",
        "|" + "---|" * (1 + len(distribution)),
    ]
    for key in next(iter(distribution.values())):
        values = [_cell(stats[key]) for stats in distribution.values()]
        lines.append(f"| {key} | " + " | ".join(values) + " |")
    return lines + [""]


def markdown(summary: dict) -> str:
    lines = benchmark_markdown(summary)
    for name, systems in summary["metrics"].items():
        lines += [f"## {name}", "", "| system | micro_f1 | macro_f1 | hmicro_f1 | p_at_1 |"]
        lines.append("|---|---|---|---|---|")
        for system, values in systems.items():
            cells = [values.get(k) for k in ("micro_f1", "macro_f1", "hmicro_f1", "p_at_1")]
            lines.append(f"| {system} | " + " | ".join(_cell(v) for v in cells) + " |")
        lines.append("")
    if "significance" in summary:
        lines += significance_markdown(summary["significance"])
    return "\n".join(lines)


def _interval(result: dict) -> str:
    return (
        f"{result['difference']:+.4f} [{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]"
        f" | {result['p_holm']:.4g}"
    )


def significance_markdown(significance: dict) -> list[str]:
    lines = [
        "## Significance",
        "",
        f"Paired project bootstrap, {significance['resamples']} resamples, 95% percentile "
        f"intervals, Holm-adjusted p. Best member on val-A: {significance['best_member']}.",
        "",
        "| test set | ensemble vs best member | micro-F1 diff [CI] | p (Holm) |",
        "|---|---|---|---|",
    ]
    for name, systems in significance["vs_best_member"].items():
        for system, result in systems.items():
            lines.append(f"| {name} | {system} | {_interval(result['micro_f1'])} |")
    lines += [
        "",
        "| system | comparison | micro-F1 diff [CI] | p (Holm) |",
        "|---|---|---|---|",
    ]
    for system, tests in significance["cross_condition"].items():
        for comparison, result in tests.items():
            lines.append(f"| {system} | {comparison} | {_interval(result['micro_f1'])} |")
    return lines + [""]


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if repeated := repeated_names(args.tune_sets, args.test_sets):
        parser().error(f"Each set name must be unique; repeated: {', '.join(repeated)}")
    families = args.members + args.baselines
    if args.bootstrap < 0:
        parser().error("--bootstrap must be a nonnegative number of resamples.")
    if len(args.members) < 2 or len(set(families)) != len(families):
        parser().error("Use at least two distinct members; baselines never vote or repeat.")
    tune_paths = {path.resolve() for _, path in args.tune_sets}
    if any(path.stem == "test" for path in tune_paths) or tune_paths & {
        path.resolve() for _, path in args.test_sets
    }:
        parser().error("Never tune on a test split or on a set that is also tested.")
    if any(name == "tuning" for name, _ in args.test_sets):
        parser().error('No test set may be named "tuning"; the label distribution uses that key.')
    predictions, output = resolve_path(args.predictions), resolve_path(args.output_dir)
    try:
        leak = leakage(args.tune_sets, args.test_sets)
        if leak:
            parser().error(f"Never tune on material that is also tested: {leak}")
        tuning = load_split(predictions, families, args.tune_sets)
        space = Space(tuning["labels"], resolve_path(args.taxonomy))
        settings = tune(space, tuning, args.members, args.baselines)
        summary = {**settings, "members": args.members, "baselines": args.baselines}
        summary |= {
            "code_commit": commit(),
            "environment": get_environment_metadata(),
            "bootstrap_seed": SEED if args.bootstrap else None,
            "tune_sets": provenance(args.tune_sets),
            "test_sets": provenance(args.test_sets),
            "threshold_grid": THRESHOLD_GRID.tolist(),
            "label_count": len(space.labels),
            "taxonomy_sha256": compute_file_sha256(resolve_path(args.taxonomy)),
            "dumps": tuning["dumps"],
            "metrics": {},
            "leave_one_out": {},
        }
        chosen: dict[str, dict] = {}
        for name, path in args.test_sets:
            data = load_split(predictions, families, [(name, path)])
            if data["labels"] != space.labels:
                raise ValueError(f"{name}: label columns differ from the tuning dumps.")
            metrics, loo, chosen[name] = evaluate_set(
                space, data, args.members, args.baselines, settings
            )
            summary["metrics"][name], summary["leave_one_out"][name] = metrics, loo
            summary["dumps"] |= data["dumps"]
        check_family_identity(summary["dumps"])
        first = args.test_sets[0][0]
        tuning_truth = binarize([r["labels_direct"] for r in tuning["rows"]], space.labels)
        scored = summary["metrics"][first]
        produced = {metric for values in scored.values() for metric in values}
        summary["benchmark"] = benchmark_systems([*scored, "leave_one_out"], produced)
        summary["benchmark"]["label_distribution"] = {
            "tuning": label_distribution(tuning_truth),
            first: label_distribution(chosen[first]["truth"]),
        }
        if args.bootstrap:
            summary["significance"] = significance(chosen, args.members, settings, args.bootstrap)
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), flush=True)
        return 1
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "summary.md").write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({"status": "success", "output": str(output)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
