"""Label coverage, zero-shot member pairs and error analysis behind the README results.

coverage  results/metrics/label_coverage.json: CORDIS codes the >= 50-project support filter
          drops without roll-up, and hierarchical F1 against the CORDIS closure within the
          active 586-node taxonomy (full unpruned-tree performance is not evaluated).
pairwise  results/metrics/cordis_v1.1.0_sample/pairwise.json: zero-shot vs trained members on
          the 120-pair sample (paired project bootstrap, Holm-adjusted within each language).
errors    results/metrics/cordis_v1.1.0_ensemble/error_analysis.json: micro-F1 by EuroSciVoc
          root, training label frequency and gold labels per project, plus the top errors.
drops     results/metrics/cordis_v1.1.0_ensemble/robustness.json: each member's EN micro-F1
          drop under TR, code-switch and noisy EN input, and SVM's drop minus each other
          member's (paired project bootstrap, Holm-adjusted within each condition).
translation  results/metrics/translation_checks.json: TR records with a possibly truncated
          aligned sentence or a repeated word n-gram the EN source lacks (data only).
mtquality results/metrics/translation_quality.json: lingua language ID of every EN and TR record
          and round-trip chrF of 300 seeded TR test records translated back to English. Run
          only when named: it downloads the pinned TR->EN model and uses CUDA when available.

CPU only, apart from mtquality. Reads the committed archive (in memory), taxonomy and splits,
and the local dumps of scripts/predict_all.py, which must be the ones whose hashes the committed
summaries recorded.
Every output of one run records the code commit taken before the first file is written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from importlib.metadata import version
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
from naltra.data.translation import HuggingFaceLocalTranslator  # noqa: E402
from naltra.evaluation.bootstrap import (  # noqa: E402
    SEED,
    f1_scores,
    holm,
    paired_bootstrap,
    resample_counts,
)
from naltra.evaluation.matrix import binarize, flat_metrics  # noqa: E402
from naltra.utils.provenance import commit, repo_path  # noqa: E402

ARCHIVE = REPO_ROOT / "data/raw/cordis_h2020/cordis-h2020projects-csv.zip"
TAXONOMY = REPO_ROOT / "taxonomy/taxonomy.json"
PROCESSED = REPO_ROOT / "data/processed/cordis_h2020"
SPLITS = {split: PROCESSED / f"en/{split}.jsonl" for split in ("train", "validation", "test")}
ENSEMBLE = REPO_ROOT / "results/metrics/cordis_v1.1.0_ensemble"
SAMPLE = REPO_ROOT / "results/metrics/cordis_v1.1.0_sample"
OUTPUTS = {
    "coverage": REPO_ROOT / "results/metrics/label_coverage.json",
    "pairwise": SAMPLE / "pairwise.json",
    "errors": ENSEMBLE / "error_analysis.json",
    "drops": ENSEMBLE / "robustness.json",
    "translation": REPO_ROOT / "results/metrics/translation_checks.json",
    "mtquality": REPO_ROOT / "results/metrics/translation_quality.json",
}
OPT_IN = ("mtquality",)  # needs the network for the model and is slow without a GPU
FULL_TEST = ("en_test", "tr_test")
ZERO_SHOT = ("jev", "kev", "laya")
TRAINED = ("transformer", "naive_bayes", "bilstm", "svm", "hybrid_knn")
ERROR_SYSTEMS = ("svm", "hybrid_knn")
# EN-train projects per direct label: tail < 70 <= mid < 200 <= head.
FREQUENCY_EDGES = (70, 200)
FREQUENCY_BUCKETS = ("tail_lt70", "mid_70_199", "head_ge200")
TOP_ERRORS = 10
REFERENCE_SET = "en_test"
DROP_CONDITIONS = ("tr_test", "cs_sentence_test", "cs_chunk_test", "en_test_noisy")
DROP_MEMBERS = ("svm", "hybrid_knn", "bilstm", "transformer", "naive_bayes")  # svm first
# The point drops recompute the committed cross_condition differences with the same code.
TOLERANCE = 1e-6
TRUNCATION_MIN_EN = 80  # characters
TRUNCATION_RATIO = 0.4
REPEAT_MIN_N = 4  # words
EXAMPLES = 5
WORD = re.compile(r"\w+")
DECIMALS = 4
BACK_MODEL = "Helsinki-NLP/opus-mt-tc-big-tr-en"
BACK_REVISION = "2261c8fc7b1af59caee87f8ff0ecf3fbccfe8391"
ROUND_TRIP_SIZE = 300
CHRF_ORDER = 6
CHRF_BETA = 2
CHRF_CUTS = (40, 30)


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
    """The CORDIS closure within the active tree: its nodes on assigned code paths."""
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
    """How far the stored labels of retained projects fall short of what CORDIS assigns.

    The node counts stay within the active tree; the unpruned path counts also include code
    paths the tree pruned (counted only, never scored).
    """
    code_of = {node: code for code, node in tree.node_by_code.items()}
    counts: Counter[str] = Counter()
    for record in records:
        assigned = codes.get(record["project_id"], set())
        dropped = assigned - tree.supported
        missing = implied_nodes(assigned, tree) - set(record["labels"])
        unpruned = {prefix for code in assigned for prefix in code_prefixes(code)} - {
            code_of[node] for node in record["labels"]
        }
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
        counts["closures_missing_unpruned_paths"] += bool(unpruned)
        counts["unpruned_paths_missing"] += len(unpruned)
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
    """Hierarchical micro-F1 of each member against the stored gold closure and the CORDIS
    closure within the active tree.

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
    """Environment and input hashes; main puts the run's code_commit in front."""
    return {
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
            "nodes_missing": "nodes of the active 586-node taxonomy on assigned code paths "
            "that the stored closure lacks",
            "unpruned_paths_missing": "code paths of assigned codes, in the active taxonomy or "
            "pruned from it, that the stored closure lacks (counted, not scored)",
            "complete": "gold closure of every assigned code within the active 586-node "
            "taxonomy, as CORDIS implies it; full unpruned-tree performance was not evaluated",
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


def interval(point: float, resampled: np.ndarray, resamples: int) -> dict[str, float]:
    """A difference with its 95% percentile interval and p-value, as paired_bootstrap gives."""
    low, high = np.percentile(resampled, [2.5, 97.5])
    tail = min(np.sum(resampled <= 0), np.sum(resampled >= 0))
    return {
        "difference": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": float(min(1.0, 2 * (1 + tail) / (1 + resamples))),
        "p_resolution": 2 / (1 + resamples),
    }


def drop_tests(reference: dict, condition: dict, resamples: int) -> dict[str, Any]:
    """Each member's micro-F1 drop from ``reference`` to ``condition`` and drop(svm) - drop(x).

    Rows are paired by pair_id. Like the committed cross_condition tests, whole projects are
    resampled with the bootstrap seed, and one set of weights serves both sets and all members.
    """
    pairs = [row["pair_id"] for row in condition["rows"]]
    index = {row["pair_id"]: i for i, row in enumerate(reference["rows"])}
    if not (len(index) == len(reference["rows"]) == len(pairs) and set(pairs) == set(index)):
        raise ValueError("The reference and condition sets must hold the same projects once.")
    order = [index[pair] for pair in pairs]
    _, member = np.unique(np.asarray(pairs), return_inverse=True)
    weights = resample_counts(len(pairs), resamples, SEED)[:, member.ravel()]
    unit = np.ones((1, len(pairs)), dtype=np.float32)
    member_drops, resampled = {}, {}
    for family in DROP_MEMBERS:
        sides = (
            (reference["truth"][order], reference["selected"][family][order]),
            (condition["truth"], condition["selected"][family]),
        )
        (point_a, point_b), (boot_a, boot_b) = (
            [f1_scores(w, truth, chosen)[0] for truth, chosen in sides] for w in (unit, weights)
        )
        resampled[family] = boot_a - boot_b
        member_drops[family] = {"reference": float(point_a[0]), "condition": float(point_b[0])}
        member_drops[family] |= interval(
            float(point_a[0] - point_b[0]), resampled[family], resamples
        )
    first, *others = DROP_MEMBERS
    tests = {
        f"{first}-{other}": interval(
            member_drops[first]["difference"] - member_drops[other]["difference"],
            resampled[first] - resampled[other],
            resamples,
        )
        for other in others
    }
    adjusted = holm({name: test["p_value"] for name, test in tests.items()})
    for name, test in tests.items():
        test["p_holm"] = adjusted[name]
    return {"projects": len(pairs), "drops": member_drops, "drop_differences": tests}


def drops(predictions: Path, resamples: int) -> dict[str, Any]:
    summary = read_summary(ENSEMBLE)
    cross = summary["significance"]["cross_condition"]
    base = load_checked(predictions, ENSEMBLE, summary, DROP_MEMBERS, REFERENCE_SET)
    conditions = {}
    for name in DROP_CONDITIONS:
        data = load_checked(predictions, ENSEMBLE, summary, DROP_MEMBERS, name)
        result = drop_tests(base, data, resamples)
        key = f"{REFERENCE_SET}->{name}"
        for family, drop in result["drops"].items():
            committed = cross[family][key]["micro_f1"]["difference"]
            if not math.isclose(drop["difference"], committed, rel_tol=0, abs_tol=TOLERANCE):
                raise ValueError(
                    f"{family} {key}: drop {drop['difference']} is not the committed {committed}."
                )
        conditions[name] = result
    return {
        **provenance([ENSEMBLE / "summary.json"]),
        "resamples": resamples,
        "bootstrap_seed": SEED,
        "reference_set": REFERENCE_SET,
        "thresholds": {family: summary["thresholds"][family] for family in DROP_MEMBERS},
        "definitions": {
            "drops": f"micro-F1 on {REFERENCE_SET} minus micro-F1 on the condition, same "
            "projects and thresholds; equals the summary's cross_condition difference",
            "drop_differences": "svm-x: drop(svm) - drop(x); above 0 means SVM loses more",
            "intervals": "paired project bootstrap over pair_id (the EN and condition copies "
            "of a project are resampled together, one draw for all members), 95% percentile "
            "interval, two-sided p with the add-one correction; p_holm is Holm-adjusted across "
            "the four drop_differences of one condition",
        },
        "conditions": conditions,
    }


def checked_split(language: str, split: str) -> Path:
    """A processed split, refused unless it is the file its committed manifest locks."""
    path = PROCESSED / language / f"{split}.jsonl"
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if compute_file_sha256(path) != manifest["output_files"][path.name]["sha256"]:
        raise ValueError(f"{language}/{path.name} differs from its manifest.")
    return path


def truncated(alignment: Iterable[Mapping[str, str]]) -> bool:
    """An aligned EN sentence over 80 characters whose TR side keeps under 40% of them."""
    return any(
        len(pair["en"]) > TRUNCATION_MIN_EN and len(pair["tr"]) < TRUNCATION_RATIO * len(pair["en"])
        for pair in alignment
    )


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def repeated_spans(tokens: Sequence[str]) -> Iterator[str]:
    """Every word n-gram (n >= 4) immediately followed by itself, as its 2n words joined."""
    for n in range(REPEAT_MIN_N, len(tokens) // 2 + 1):
        run = 0  # consecutive positions j with tokens[j] == tokens[j + n]
        for j in range(len(tokens) - n):
            run = run + 1 if tokens[j] == tokens[j + n] else 0
            if run >= n:
                yield " ".join(tokens[j - n + 1 : j + n + 1])


def mt_repetition(text: str, source: str) -> bool:
    """A repeated n-gram of the TR text whose doubled form the EN source does not contain."""
    padded = f" {' '.join(words(source))} "
    return any(f" {span} " not in padded for span in repeated_spans(words(text)))


def flag_rates(flagged: Mapping[str, list[str]], records: int) -> dict[str, Any]:
    return {"records": records} | {
        check: {"records": len(ids), "rate": len(ids) / records, "examples": ids[:EXAMPLES]}
        for check, ids in flagged.items()
    }


def screened(records: Sequence[Mapping], sources: Mapping[str, str]) -> dict[str, list[str]]:
    """Ids of the TR records each screen flags, in file order; sources are EN texts by pair_id."""
    return {
        "truncation": [r["id"] for r in records if truncated(r["sentence_alignment"])],
        "repetition": [r["id"] for r in records if mt_repetition(r["text"], sources[r["pair_id"]])],
    }


def translation_checks() -> dict[str, Any]:
    splits, inputs, totals = {}, [], Counter()
    for split in SPLITS:
        en, tr = checked_split("en", split), checked_split("tr", split)
        sources = {record["pair_id"]: record["text"] for record in load_jsonl(en)}
        records = load_jsonl(tr)
        flagged = screened(records, sources)
        splits[split] = flag_rates(flagged, len(records))
        totals.update({"records": len(records)} | {k: len(v) for k, v in flagged.items()})
        inputs += [en, tr]
    splits["all"] = {"records": totals["records"]} | {
        check: {"records": totals[check], "rate": totals[check] / totals["records"]}
        for check in ("truncation", "repetition")
    }
    return {
        **provenance(inputs),
        "definitions": {
            "truncation": "a TR record with at least one sentence_alignment pair whose EN "
            f"sentence is longer than {TRUNCATION_MIN_EN} characters and whose TR sentence has "
            f"fewer than {TRUNCATION_RATIO:.0%} of its characters (possible truncation)",
            "repetition": "a TR record whose text, lowercased and split into \\w+ words, holds "
            f"a word n-gram (n >= {REPEAT_MIN_N}) immediately followed by itself, where that "
            "doubled 2n-word sequence does not occur in the paired EN split text split the "
            "same way (possible MT repetition)",
            "rate": "flagged TR records / TR records in the split",
            "examples": f"the first {EXAMPLES} flagged TR record ids in file order",
            "scope": "automatic screens: flagged records are candidates, not confirmed errors, "
            "and the released text is unchanged",
        },
        "splits": splits,
    }


def language_counts(
    records: Iterable[Mapping[str, Any]], expected: str, detect: Callable[[str], str]
) -> dict[str, Any]:
    """Detected language of every record's text, and the records not detected as ``expected``."""
    found = {record["id"]: detect(record["text"]) for record in records}
    other = [record_id for record_id, language in found.items() if language != expected]
    detected = dict(sorted(Counter(found.values()).items()))
    return {"records": len(found), "detected": detected} | flag_rates(
        {f"not_{expected}": other}, len(found)
    )


def language_id(language: str, detect: Callable[[str], str]) -> dict[str, Any]:
    """language_counts of each manifest-locked split of ``language``, and of all of them."""
    splits = {
        split: language_counts(load_jsonl(checked_split(language, split)), language, detect)
        for split in SPLITS
    }
    detected = sum((Counter(counts["detected"]) for counts in splits.values()), Counter())
    records = detected.total()
    splits["all"] = {
        "records": records,
        "detected": dict(sorted(detected.items())),
        f"not_{language}": {
            "records": records - detected[language],
            "rate": (records - detected[language]) / records,
        },
    }
    return splits


def char_ngrams(text: str, n: int) -> Counter[str]:
    chars = "".join(text.split())
    return Counter(chars[i : i + n] for i in range(len(chars) - n + 1))


def chrf(hypothesis: str, reference: str) -> float:
    """chrF (Popovic 2015) on 0-100: character n-grams of orders 1..6, whitespace removed.

    Clipped n-gram precision and recall are each averaged over the orders both texts reach and
    combined as F-beta with beta = 2, so recall counts more than precision.
    """
    precision = recall = 0.0
    orders = 0
    for n in range(1, CHRF_ORDER + 1):
        hyp, ref = char_ngrams(hypothesis, n), char_ngrams(reference, n)
        if hyp and ref:
            matches = (hyp & ref).total()
            precision += matches / hyp.total()
            recall += matches / ref.total()
            orders += 1
    if not precision + recall:
        return 0.0
    precision, recall, factor = precision / orders, recall / orders, CHRF_BETA**2
    return 100 * (1 + factor) * precision * recall / (factor * precision + recall)


def round_trip_sample(
    ids: Iterable[str], size: int = ROUND_TRIP_SIZE, seed: int = SEED
) -> list[str]:
    """``size`` ids drawn with the seed from their sorted order (not file order), sorted."""
    return sorted(random.Random(seed).sample(sorted(ids), size))


def round_trip_summary(
    scores: Mapping[str, float], sample: Sequence[str], flags: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    """chrF distribution of the sample; mean chrF of screened records and the unflagged sample."""
    values = np.array([scores[record_id] for record_id in sample])
    q1, median, q3 = (float(q) for q in np.percentile(values, [25, 50, 75]))
    in_sample, flagged = set(sample), set().union(*flags.values())

    def group(ids: Iterable[str]) -> dict[str, Any]:
        ids = sorted(ids)
        return {
            "documents": len(ids),
            "in_sample": len(in_sample.intersection(ids)),
            "mean_chrf": float(np.mean([scores[i] for i in ids])) if ids else None,
        }

    lowest = sorted(sample, key=lambda record_id: (scores[record_id], record_id))[:EXAMPLES]
    below = {cut: values < cut for cut in CHRF_CUTS}
    return {
        "chrf": {
            "mean": float(values.mean()),
            "q1": q1,
            "median": median,
            "q3": q3,
            "min": float(values.min()),
            "max": float(values.max()),
        },
        **{
            f"below_{cut}": {"documents": int(mask.sum()), "rate": float(mask.mean())}
            for cut, mask in below.items()
        },
        "screens": {name: group(ids) for name, ids in flags.items()}
        | {"any_screen": group(flagged), "unflagged_sample": group(in_sample - flagged)},
        "lowest": {record_id: scores[record_id] for record_id in lowest},
    }


class BackTranslator(HuggingFaceLocalTranslator):
    """The release translator's chunking, batching and decoding on the pinned TR->EN model."""

    def __init__(self, device: str | None, batch_size: int) -> None:
        super().__init__(BACK_MODEL, device=device, batch_size=batch_size)
        self._revision = BACK_REVISION

    def settings(self) -> dict[str, Any]:
        self._load_model()
        return {
            "model": self.model_name,
            "revision": self.model_revision,
            "device": self._device,
            "dtype": str(next(self._model.parameters()).dtype),
            "batch_size": self._batch_size,
            "num_beams": self._model.generation_config.num_beams,
            "no_repeat_ngram_size": 4,  # as HuggingFaceLocalTranslator.translate_sentences
            "max_new_tokens": self._max_length,
        }


def round_trip(device: str | None, batch_size: int) -> dict[str, Any]:
    """Back-translate the seeded sample and every screened TR test record; score them by chrF."""
    sources = {r["pair_id"]: r["text"] for r in load_jsonl(checked_split("en", "test"))}
    records = load_jsonl(checked_split("tr", "test"))
    flags = screened(records, sources)
    sample = round_trip_sample(record["id"] for record in records)
    chosen = set(sample).union(*flags.values())
    documents = [record for record in records if record["id"] in chosen]
    translator = BackTranslator(device, batch_size)
    settings = translator.settings()
    back = translator.translate_sentences(
        [pair["tr"] for record in documents for pair in record["sentence_alignment"]]
    )
    scores, start = {}, 0
    for record in documents:
        end = start + len(record["sentence_alignment"])
        scores[record["id"]] = chrf(" ".join(back[start:end]), sources[record["pair_id"]])
        start = end
    return {
        "translator": settings,
        "sample": {
            "split": "test",
            "population": len(records),
            "documents": len(sample),
            "seed": SEED,
            "scored_documents": len(documents),
        },
        **round_trip_summary(scores, sample, flags),
    }


def mt_quality(device: str | None, batch_size: int) -> dict[str, Any]:
    from naltra.pipeline.language import LinguaLanguageDetector

    detector = LinguaLanguageDetector()
    inputs = [PROCESSED / lang / f"{split}.jsonl" for lang in ("en", "tr") for split in SPLITS]
    return {
        **provenance(inputs),
        "tools": {name: version(name) for name in ("lingua-language-detector", "sentencepiece")},
        "definitions": {
            "language_id": "primary language of each record's text by naltra.pipeline.language."
            "LinguaLanguageDetector (lingua restricted to EN and TR; minimum confidence "
            f"{detector.minimum_confidence}, minimum share {detector.minimum_share}): en-tr when "
            "both languages hold at least 2 words and that share of the confidently detected "
            "words, und under 2 words or below that confidence",
            "not_tr, not_en": "records not detected as their own language; examples are the "
            f"first {EXAMPLES} in file order",
            "round_trip": f"{ROUND_TRIP_SIZE} TR test records drawn by random.Random(seed).sample "
            "from the sorted record ids, plus every TR test record a translation_checks screen "
            "flags; each stored TR aligned sentence is translated back to English with the "
            "pinned model and the sentences are joined with spaces",
            "chrf": "document chrF (Popovic 2015) of the back-translation against the EN source "
            f"text: character n-grams of orders 1..{CHRF_ORDER} with whitespace removed, clipped "
            "matches, precision and recall averaged over the orders, F-beta with beta = "
            f"{CHRF_BETA}, 0-100; chrf, below_40, below_30 and lowest cover the sample only",
            "screens": "mean chrF of every TR test record each translation_checks screen flags "
            "(in_sample: how many are in the sample) and of the sampled records no screen flags",
            "scope": "round-trip chrF is an automatic proxy, not a human judgement: errors of "
            "the EN->TR release model can cancel out or be introduced by the reverse model, and "
            "chrF measures character overlap with the EN source, not Turkish fluency; language "
            "ID shows a record reads as Turkish, not that it is a faithful translation",
        },
        "language_id": {
            language: language_id(language, lambda text: detector.detect(text).primary)
            for language in ("tr", "en")
        },
        "round_trip": round_trip(device, batch_size),
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
    result.add_argument(
        "analyses", nargs="*", choices=sorted(OUTPUTS), help=f"default: all but {', '.join(OPT_IN)}"
    )
    result.add_argument("--predictions", default="results/predictions/cordis_v1.1.0")
    result.add_argument("--resamples", type=int, default=10000, help="Paired bootstrap resamples")
    result.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto", help="mtquality model device"
    )
    result.add_argument("--batch-size", type=int, default=8, help="mtquality sentences per batch")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.resamples < 1 or args.batch_size < 1:
        parser().error("--resamples and --batch-size must be positive.")
    predictions = resolve_path(args.predictions)
    builders: dict[str, Callable[[], dict]] = {
        "coverage": lambda: label_coverage(predictions),
        "pairwise": lambda: pairwise(predictions, args.resamples),
        "errors": lambda: error_analysis(predictions),
        "drops": lambda: drops(predictions, args.resamples),
        "translation": lambda: translation_checks(),
        "mtquality": lambda: mt_quality(
            None if args.device == "auto" else args.device, args.batch_size
        ),
    }
    # Once, before any write: a written result would make later outputs record a dirty tree.
    code_commit = commit()
    default = [name for name in sorted(OUTPUTS) if name not in OPT_IN]
    for name in dict.fromkeys(args.analyses or default):
        try:
            payload = {"code_commit": code_commit, **builders[name]()}
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
