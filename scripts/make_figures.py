"""Render the paper figures from evaluate_ensemble's summary.json and the prediction dumps.

Figures: micro-F1 per test language (members and ensembles), root-domain errors,
micro-F1 by hierarchy depth, and top-1 reliability. Thresholds are the val-A thresholds
stored in the summary. Every dump goes through the benchmark's validated loader and must be
the exact dump, scored on the same gold file, that the summary recorded, so the figures match
the reported tables. Nothing is written unless every input validates.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.evaluate_ensemble import load_split  # noqa: E402
from scripts.predict_all import parse_set, repeated_names, resolve_path  # noqa: E402

from naltra.data.manifest import compute_file_sha256  # noqa: E402
from naltra.evaluation.figures import depth_f1, reliability_bins, root_confusion  # noqa: E402
from naltra.evaluation.matrix import binarize, close_upward  # noqa: E402

# Validated categorical slots (fixed order, light mode) and chart chrome.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BOUND = ("npz_sha256", "input_sha256")
SHORT_ROOTS = {
    "medical_and_health_sciences": "medical & health",
    "natural_sciences": "natural",
    "engineering_and_technology": "engineering & tech",
    "agricultural_sciences": "agricultural",
    "social_sciences": "social",
    "humanities": "humanities",
}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "savefig.dpi": 200,
        }
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--summary", required=True, help="evaluate_ensemble summary.json")
    result.add_argument("--predictions", required=True)
    result.add_argument("--test-sets", nargs="+", type=parse_set, required=True)
    result.add_argument("--output-dir", default="results/plots")
    result.add_argument("--taxonomy", default=str(REPO_ROOT / "taxonomy/taxonomy.json"))
    return result


def language_f1(summary: dict, target: Path) -> None:
    sets = list(summary["metrics"])
    systems = [
        s for s in summary["metrics"][sets[0]] if "micro_f1" in summary["metrics"][sets[0]][s]
    ]
    width = 0.8 / len(systems)
    figure, axis = plt.subplots(figsize=(8, 4))
    for index, system in enumerate(systems):
        values = [summary["metrics"][name][system]["micro_f1"] for name in sets]
        offsets = np.arange(len(sets)) - 0.4 + width * (index + 0.5)
        axis.bar(offsets, values, width * 0.9, color=SERIES[index % len(SERIES)], label=system)
    axis.set_xticks(np.arange(len(sets)), sets)
    axis.set_ylabel("micro-F1 (473 direct labels)")
    axis.set_ylim(0, 1)
    axis.grid(axis="x", visible=False)
    axis.legend(ncols=min(4, len(systems)), loc="upper right", fontsize=8)
    axis.set_title("Micro-F1 by test language")
    figure.tight_layout()
    figure.savefig(target)
    plt.close(figure)


def confusion_panels(panels: list[tuple[str, list[str], np.ndarray]], target: Path) -> None:
    figure, axes = plt.subplots(
        1, len(panels), figsize=(3.9 * len(panels) + 1.6, 4.2), squeeze=False, sharey=True
    )
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    for axis, (title, roots, matrix) in zip(axes[0], panels, strict=True):
        image = axis.imshow(matrix, cmap=cmap, vmin=0, vmax=1)
        labels = [SHORT_ROOTS.get(root, root.replace("_", " ")) for root in roots]
        axis.set_xticks(range(len(roots)), labels, rotation=40, ha="right", fontsize=8)
        axis.grid(False)
        for (row, column), value in np.ndenumerate(matrix):
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=SURFACE if value > 0.55 else INK,
            )
        axis.set_xlabel("predicted root")
        axis.set_title(title)
    axes[0][0].set_yticks(range(len(labels)), labels, fontsize=8)
    axes[0][0].set_ylabel("gold root")
    figure.colorbar(
        image,
        ax=axes[0].tolist(),
        shrink=0.8,
        label="diagonal: recall; off-diagonal: false root rate",
    )
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)


def line_panel(
    series: dict[str, tuple],
    xlabel: str,
    ylabel: str,
    title: str,
    target: Path,
    diagonal: bool = False,
) -> None:
    figure, axis = plt.subplots(figsize=(6, 4))
    if diagonal:
        axis.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="perfect")
    for index, (name, (x, y)) in enumerate(series.items()):
        axis.plot(x, y, color=SERIES[index % len(SERIES)], marker="o", markersize=5, label=name)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_ylim(0, 1)
    axis.set_title(title)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(target)
    plt.close(figure)


def scored_split(
    summary: dict, predictions: Path, families: list[str], test_set: tuple[str, Path]
) -> dict:
    """Load one set through the benchmark's validated loader, bound to the dumps it scored."""
    data = load_split(predictions, families, [test_set])
    for key, origin in data["dumps"].items():
        recorded = summary.get("dumps", {}).get(key)
        if not isinstance(recorded, dict) or any(recorded.get(f) != origin[f] for f in BOUND):
            raise ValueError(f"{key}: not the dump the summary scored; re-run evaluate_ensemble.")
    return data


def collect(args: argparse.Namespace, summary: dict) -> tuple[list, dict, dict]:
    """Validate every input and compute all figure data without writing anything."""
    taxonomy_path = resolve_path(args.taxonomy)
    if compute_file_sha256(taxonomy_path) != summary.get("taxonomy_sha256"):
        raise ValueError(
            "Taxonomy differs from the one the summary used; re-run evaluate_ensemble."
        )
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    parents = {item["id"]: item["parent"] for item in taxonomy["labels"]}
    nodes = [item["id"] for item in taxonomy["labels"]]
    families = summary["members"] + summary.get("baselines", [])
    predictions = resolve_path(args.predictions)
    panels, depth_series, reliability = [], {}, {}
    for name, path in args.test_sets:
        data = scored_split(summary, predictions, families, (name, path))
        labels = data["labels"]
        truth = binarize([r["labels_direct"] for r in data["rows"]], labels)
        gold_closed = close_upward(truth, labels, nodes, parents)
        for family in families:
            scores = data["scores"][family]
            selected = scores >= summary["thresholds"][family]
            predicted_closed = close_upward(selected, labels, nodes, parents)
            if family == families[0]:
                roots, matrix = root_confusion(gold_closed, predicted_closed, nodes, parents)
                panels.append((f"{family} - {name}", roots, matrix))
            if name == args.test_sets[0][0]:
                levels = depth_f1(gold_closed, predicted_closed, nodes, parents)
                depth_series[family] = (list(levels), list(levels.values()))
                bins = reliability_bins(truth, scores)
                reliability[family] = (bins["confidence"], bins["accuracy"])
    return panels, depth_series, reliability


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if repeated := repeated_names(args.test_sets):
        parser().error(f"Each set name must be unique; repeated: {', '.join(repeated)}")
    style()
    try:
        summary = json.loads(resolve_path(args.summary).read_text(encoding="utf-8"))
        panels, depth_series, reliability = collect(args, summary)
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), flush=True)
        return 1
    output = resolve_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    language_f1(summary, output / "language_f1.png")
    confusion_panels(panels, output / "root_confusion.png")
    first = args.test_sets[0][0]
    line_panel(
        depth_series,
        "depth in EuroSciVoc",
        "micro-F1 (closed space)",
        f"Micro-F1 by hierarchy depth ({first})",
        output / "depth_f1.png",
    )
    line_panel(
        reliability,
        "top-1 confidence",
        "top-1 accuracy",
        f"Top-1 reliability ({first})",
        output / "reliability.png",
        diagonal=True,
    )
    print(json.dumps({"status": "success", "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
