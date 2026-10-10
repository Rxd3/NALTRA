"""Build the interactive NALTRA results page from a benchmark summary and its dumps.

Every dump is loaded through the benchmark's validated loader and must be one the summary
scored, so the page shows exactly the reported numbers; optional latency and OOD summaries
must have run on the same test files, and an optional EN/TR sample summary (the only scores
of Kev and Jev) on the same taxonomy. Writes a standalone HTML page
(GitHub Pages / local) and, optionally, a fragment for hosted publishing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.evaluate_ensemble import Space, evaluate_set  # noqa: E402
from scripts.make_figures import scored_split  # noqa: E402
from scripts.predict_all import parse_set, repeated_names, resolve_path  # noqa: E402

from naltra.data.manifest import compute_file_sha256  # noqa: E402
from naltra.evaluation.figures import depth_f1, reliability_bins, root_confusion  # noqa: E402
from naltra.evaluation.offline import weighted_soft_vote  # noqa: E402
from naltra.utils.provenance import commit  # noqa: E402

TEMPLATE = REPO_ROOT / "src/naltra/reporting/dashboard.html"
PLACEHOLDER = "/*__NALTRA_DATA__*/null"
SETTINGS = ("thresholds", "val_a_micro_f1", "k", "k_closed", "soft_threshold")
ARTIFACT = ("artifact_sha256", "artifact_files_sha256")
OOD_COUNTS = ("id_records", "ood_records")
OOD_RATES = ("auroc", "fpr_at_95_tpr")
OOD_INTERVAL = ("auroc_ci_low", "auroc_ci_high")
NOTES = [
    "Data: CORDIS Horizon 2020 project objectives labelled with EuroSciVoc; English source, "
    "Turkish machine translation and synthetic English-Turkish code-switching of the same "
    "projects, plus noisy copies of the English and Turkish test sets.",
    "Validation is split by project (pair_id) into two halves: half A tunes each member's "
    "decision threshold, half B tunes the ensemble vote count and soft-vote thresholds. "
    "Test sets are scored once with those frozen settings.",
    "Micro- and macro-F1 cover all 473 directly supported labels; hierarchical scores close "
    "predictions and gold labels upward through the 586-node taxonomy.",
    "Intervals come from a paired bootstrap over projects with Holm correction per family of "
    "comparisons; p-values use an add-one correction, so the smallest reportable p is "
    "2 / (resamples + 1).",
    "Labels are CORDIS's semi-automatic assignments (silver standard) and the Turkish text is "
    "machine-translated without human validation.",
]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--summary", required=True, help="evaluate_ensemble summary.json")
    result.add_argument("--predictions", required=True)
    result.add_argument("--test-sets", nargs="+", type=parse_set, required=True)
    result.add_argument("--latency", help="run_benchmark benchmark_summary.json")
    result.add_argument("--ood", help="run_ood summary.json")
    result.add_argument(
        "--sample-summary", help="evaluate_ensemble summary.json of the EN/TR test sample"
    )
    result.add_argument("--taxonomy", default=str(REPO_ROOT / "taxonomy/taxonomy.json"))
    result.add_argument("--output", default="docs/results/index.html")
    result.add_argument("--fragment", help="Also write the page without the HTML skeleton")
    return result


def system_detail(space: Space, truth: np.ndarray, selected: np.ndarray, scores=None) -> dict:
    gold, predicted = space.close(truth), space.close(selected)
    roots, matrix = root_confusion(gold, predicted, space.nodes, space.parents)
    root_columns = [space.nodes.index(root) for root in roots]
    detail: dict[str, Any] = {
        "confusion": {
            "roots": roots,
            "matrix": np.round(matrix, 4).tolist(),
            "support": gold[:, root_columns].sum(axis=0).astype(int).tolist(),
        },
        "depth": {
            str(k): round(v, 4)
            for k, v in depth_f1(gold, predicted, space.nodes, space.parents).items()
        },
    }
    if scores is not None:
        bins = reliability_bins(truth, scores)
        detail["reliability"] = {
            "confidence": np.round(bins["confidence"], 4).tolist(),
            "accuracy": np.round(bins["accuracy"], 4).tolist(),
            "count": bins["count"].astype(int).tolist(),
        }
    return detail


def set_detail(summary: dict, space: Space, data: dict) -> dict[str, dict]:
    members, baselines = summary["members"], summary.get("baselines", [])
    settings = {key: summary[key] for key in SETTINGS} | {
        "weighted_soft_threshold": summary["weighted_soft_threshold"]
    }
    _, _, chosen = evaluate_set(space, data, members, baselines, settings)
    stack = np.stack([data["scores"][family] for family in members])
    weights = np.array([summary["val_a_micro_f1"][family] for family in members])
    probabilities = {family: data["scores"][family] for family in members + baselines}
    probabilities |= {
        "soft": stack.mean(axis=0),
        "weighted_soft": weighted_soft_vote(stack, weights),
    }
    return {
        system: system_detail(space, chosen["truth"], selected, probabilities.get(system))
        for system, selected in chosen["systems"].items()
    }


def read_json(value: str | None) -> Any:
    """The parsed file, or None without a path; a file holding only null is refused."""
    if not value:
        return None
    data = json.loads(resolve_path(value).read_text(encoding="utf-8"))
    if data is None:
        raise ValueError(f"{Path(value).name} holds null instead of a summary.")
    return data


def finite(value: Any, low: float = -math.inf, high: float = math.inf) -> bool:
    """Whether a value is a finite number (not a bool) in [low, high]."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        number = float(value)  # an integer too large for a float is not a usable number
    except OverflowError:
        return False
    return math.isfinite(number) and low <= number <= high


def numbers(
    record: Any,
    required: Sequence[str],
    optional: Sequence[str] = (),
    low: float = -math.inf,
    high: float = math.inf,
) -> bool:
    """Whether a record holds every required field, and any optional one, as a finite number
    in [low, high]."""
    return (
        isinstance(record, dict)
        and all(finite(record.get(key), low, high) for key in required)
        and all(record.get(key) is None or finite(record[key], low, high) for key in optional)
    )


def check_latency(summary: dict, tested: dict, latency: Any) -> None:
    if not (
        isinstance(latency, dict)
        and isinstance(latency.get("eval_set"), str)
        and isinstance(latency.get("hardware"), dict)
        and isinstance(latency.get("models"), dict)
        and latency["models"]
        and numbers(latency, (), ("samples",), low=0)
        and all(
            numbers(timing, (), ("artifact_bytes",), low=0)
            and numbers(timing.get("single"), ("p50_ms", "p95_ms"), low=0)
            and numbers(timing.get("batched"), ("docs_per_second",), low=0)
            for timing in latency["models"].values()
        )
    ):
        raise ValueError("Latency summary is malformed; re-run run_benchmark.")
    if (latency["eval_set"], latency.get("eval_set_sha256")) not in tested.items():
        raise ValueError("Latency was timed on another test set; re-run run_benchmark.")
    for model, timing in latency["models"].items():
        scored = summary.get("dumps", {}).get(f"{model}/{latency['eval_set']}")
        if not (scored and all(timing.get(key) == scored.get(key) for key in ARTIFACT)):
            raise ValueError(
                f"Latency of {model} was timed on other artifacts than the summary scored; "
                "re-run run_benchmark."
            )


def check_ood(summary: dict, tested: dict, ood: Any) -> None:
    if not (
        isinstance(ood, dict)
        and isinstance(ood.get("inputs"), dict)
        and isinstance(ood.get("counts"), dict)
        and isinstance(ood.get("results"), dict)
        and ood["results"]
        and all(
            isinstance(result, dict)
            and all(
                numbers(scores, OOD_COUNTS, low=0)
                and numbers(scores, OOD_RATES, OOD_INTERVAL, low=0, high=1)
                for language, scores in result.items()
                if language != "trained_labels"
            )
            for result in ood["results"].values()
        )
    ):
        raise ValueError("OOD summary is malformed; re-run run_ood.")
    if ood.get("taxonomy_sha256") != summary.get("taxonomy_sha256"):
        raise ValueError("OOD results used another taxonomy than the summary; re-run run_ood.")
    recorded = {
        key.replace("/", "_"): digest
        for key, digest in ood["inputs"].items()
        if key.endswith("/test")
    }
    if not recorded or not recorded.items() <= tested.items():
        raise ValueError("OOD results score other test sets than the summary; re-run run_ood.")
    languages = {name.removesuffix("_test") for name in recorded}
    for model, result in ood["results"].items():
        if result.keys() - {"trained_labels"} != languages:
            raise ValueError(
                f"OOD results of {model} cover other languages than the recorded test inputs; "
                "re-run run_ood."
            )


def lookup(record: Any, *keys: str) -> Any:
    """record[key][key]..., or None where a level is missing or not an object."""
    for key in keys:
        record = record.get(key) if isinstance(record, dict) else None
    return record


def sample_results(summary: dict, sample: Any) -> dict:
    """Validate an EN/TR sample summary and keep only what the page's sample table shows."""
    if not (
        isinstance(sample, dict)
        and isinstance(sample.get("test_sets"), dict)
        and isinstance(sample.get("metrics"), dict)
        and isinstance(sample.get("members"), list)
        and all(isinstance(member, str) for member in sample["members"])
    ):
        raise ValueError("Sample summary is malformed; re-run evaluate_ensemble.")
    if sample.get("taxonomy_sha256") != summary.get("taxonomy_sha256"):
        raise ValueError(
            "Sample summary used another taxonomy than the summary; re-run evaluate_ensemble."
        )
    names = {name.split("_test", 1)[0]: name for name in sample["test_sets"] if "_test" in name}
    if len(sample["test_sets"]) != 2 or names.keys() != {"en", "tr"}:
        raise ValueError("Sample summary must score one English and one Turkish test set.")
    digests = [lookup(sample, "test_sets", name, "sha256") for name in names.values()]
    if not all(isinstance(digest, str) for digest in digests) or digests[0] == digests[1]:
        raise ValueError(
            "Sample English and Turkish test sets must be two different hashed files; "
            "re-run evaluate_ensemble."
        )
    benchmark = summary.get("benchmark") or {}
    headline = [
        benchmark.get("primary_metric", "micro_f1"),
        *benchmark.get("secondary_metrics", []),
    ]
    scores = {language: sample["metrics"].get(name) for language, name in names.items()}
    if not (
        all(isinstance(systems, dict) and systems for systems in scores.values())
        and scores["en"].keys() == scores["tr"].keys()
        and all(
            numbers(m, (), headline, low=0, high=1) and any(finite(m.get(key)) for key in headline)
            for systems in scores.values()
            for m in systems.values()
        )
    ):
        raise ValueError("Sample metrics are malformed; re-run evaluate_ensemble.")
    pair, drops = f"{names['en']}->{names['tr']}", {}
    for system in scores["en"]:
        drop = lookup(sample, "significance", "cross_condition", system, pair, "micro_f1")
        if drop is not None:
            if not numbers(drop, ("difference", "ci_low", "ci_high")):
                raise ValueError("Sample significance is malformed; re-run evaluate_ensemble.")
            drops[system] = {key: drop[key] for key in ("difference", "ci_low", "ci_high")}
    shown = {
        language: {
            system: {key: m[key] for key in headline if m.get(key) is not None}
            for system, m in systems.items()
        }
        for language, systems in scores.items()
    }
    records = lookup(sample, "benchmark", "label_distribution", names["en"], "n_records")
    if records is not None and not finite(records, low=0):
        raise ValueError("Sample record count is malformed; re-run evaluate_ensemble.")
    roles = {s: lookup(sample, "benchmark", "systems", s, "role") for s in scores["en"]}
    return {
        "records": records,
        "members": sample["members"],
        "headline": headline,
        **shown,
        "drop": drops,
        "roles": {system: role for system, role in roles.items() if isinstance(role, str)},
    }


def check_release(summary: dict, latency: Any, ood: Any) -> None:
    """Refuse a malformed summary, and latency or OOD results from other test files,
    artifacts or taxonomy."""
    entries = summary.get("test_sets")
    if not (isinstance(entries, dict) and all(isinstance(e, dict) for e in entries.values())):
        raise ValueError("Summary test sets are malformed; re-run evaluate_ensemble.")
    benchmark = summary.get("benchmark") or {}
    if not isinstance(benchmark, dict):
        raise ValueError("Summary benchmark is not an object; re-run evaluate_ensemble.")
    primary = benchmark.get("primary_metric", "micro_f1")
    secondary = benchmark.get("secondary_metrics", [])
    if not (
        isinstance(primary, str)
        and isinstance(secondary, list)
        and all(isinstance(key, str) for key in secondary)
    ):
        raise ValueError("Summary benchmark metrics are malformed; re-run evaluate_ensemble.")
    metrics = summary.get("metrics")
    if not (
        isinstance(metrics, dict)
        and all(
            isinstance(systems, dict)
            and all(numbers(m, (), [primary, *secondary], low=0, high=1) for m in systems.values())
            for systems in metrics.values()
        )
    ):
        raise ValueError(
            "Summary metrics are malformed or outside [0, 1]; re-run evaluate_ensemble."
        )
    tested = {name: entry["sha256"] for name, entry in entries.items()}
    if latency is not None:
        check_latency(summary, tested, latency)
    if ood is not None:
        check_ood(summary, tested, ood)


def collect(args: argparse.Namespace) -> dict[str, Any]:
    """Validate every input and assemble the page data; writes nothing."""
    summary = json.loads(resolve_path(args.summary).read_text(encoding="utf-8"))
    taxonomy_path = resolve_path(args.taxonomy)
    if compute_file_sha256(taxonomy_path) != summary.get("taxonomy_sha256"):
        raise ValueError(
            "Taxonomy differs from the one the summary used; re-run evaluate_ensemble."
        )
    latency, ood = read_json(args.latency), read_json(args.ood)
    check_release(summary, latency, ood)
    sample = read_json(args.sample_summary)
    sample = None if sample is None else sample_results(summary, sample)
    families = summary["members"] + summary.get("baselines", [])
    predictions = resolve_path(args.predictions)
    detail, space, records = {}, None, 0
    for name, path in args.test_sets:
        data = scored_split(summary, predictions, families, (name, path))
        space = space or Space(data["labels"], taxonomy_path)
        detail[name] = set_detail(summary, space, data)
        records = records or len(data["rows"])
    names = [name for name, _ in args.test_sets]
    significance = summary.get("significance")
    best = (
        significance["best_member"]
        if significance
        else max(summary["members"], key=lambda family: summary["val_a_micro_f1"][family])
    )
    voting = len(summary["members"])
    pending = (
        f"Only {voting} voting members: with fewer than three voters a strict majority needs "
        "every member to agree, so the hard-vote ensembles are degenerate (unanimity or union)."
        if voting < 3
        else None
    )
    if latency:
        latency.pop("eval_set_path", None)  # a local machine path; eval_set + sha256 identify it
    if ood:
        for result in ood["results"].values():
            result.pop("trained_labels", None)
    return {
        "meta": {
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "commit": commit() or "unknown",
            "labels": summary["label_count"],
            "nodes": len(space.nodes),
            "members": summary["members"],
            "baselines": summary.get("baselines", []),
            "best_member": best,
            "thresholds": summary["thresholds"],
            "test_sets": names,
            "test_records": records,
            "pending": pending,
            "notes": NOTES,
        },
        "metrics": {name: summary["metrics"][name] for name in names},
        "leave_one_out": {name: summary["leave_one_out"].get(name, {}) for name in names},
        "significance": significance,
        "benchmark": summary.get("benchmark"),
        "detail": detail,
        "latency": latency,
        "ood": ood,
        "sample": sample,
    }


def render(data: dict[str, Any]) -> tuple[str, str]:
    """Return (standalone page, fragment); embedded JSON cannot close the script element."""
    # "<" only occurs inside JSON strings; < keeps "</script>" and "<!--" inert.
    payload = json.dumps(data, allow_nan=False).replace("<", "\\u003c")
    fragment = TEMPLATE.read_text(encoding="utf-8").replace(PLACEHOLDER, payload)
    head, body = fragment.split('<div class="wrap">', 1)
    page = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        f'{head}</head>\n<body>\n<div class="wrap">{body}</body>\n</html>\n'
    )
    return page, fragment


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if repeated := repeated_names(args.test_sets):
        parser().error(f"Each set name must be unique; repeated: {', '.join(repeated)}")
    try:
        page, fragment = render(collect(args))
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), flush=True)
        return 1
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    if args.fragment:
        target = resolve_path(args.fragment)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(fragment, encoding="utf-8")
    print(json.dumps({"status": "success", "output": str(output)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
