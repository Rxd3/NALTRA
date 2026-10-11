"""Render release-summary charts without model artifacts or prediction matrices.

These figures visualize archived measurements; they do not rerun any experiment.
Run from the repository root. Inputs must use the release summary schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "svm": "SVM",
    "hybrid_knn": "Hybrid kNN",
    "bilstm": "BiLSTM",
    "transformer": "XLM-R",
    "naive_bayes": "Naive Bayes",
    "kev": "Kev",
    "jev": "Jev",
    "laya": "Laya",
    "hard_majority": "Hard majority",
    "hard_k": "Tuned k-of-M",
    "soft": "Soft vote",
    "weighted_soft": "Weighted soft vote",
}
CONDITIONS = {
    "en_test": "English",
    "tr_test": "Turkish",
    "cs_chunk_test": "Chunk\nmix",
    "cs_sentence_test": "Sentence\nmix",
    "en_test_noisy": "English\nnoisy",
    "tr_test_noisy": "Turkish\nnoisy",
}
MEMBERS = ("svm", "naive_bayes", "bilstm", "transformer", "hybrid_knn")
BLUE, ORANGE, INK, MUTED = "#2878b5", "#d47722", "#182b3a", "#536574"
FIGURES = (
    "full_f1",
    "robustness_heatmap",
    "robustness_drops",
    "sample_f1",
    "cpu_benchmark",
    "accuracy_latency",
    "artifact_size",
    "ood_detection",
)
DEFAULTS = {
    "full": "results/metrics/cordis_v1.1.0_ensemble/summary.json",
    "sample": "results/metrics/cordis_v1.1.0_sample/summary.json",
    "benchmark": "results/benchmarks/cordis_v1.1.0/benchmark_summary.json",
    "ood": "results/ood/cordis_v1.1.0/summary.json",
}


def number(value: object, context: str, low: float = 0, high: float = 1) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}: expected a number")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{context}: expected a finite value in [{low}, {high}]")
    return float(value)


def systems(summary: dict, dataset: str) -> list[str]:
    # The hierarchy-closed vote has no direct-label F1 and is deliberately omitted.
    rows = summary["metrics"][dataset]
    names = [name for name, row in rows.items() if "micro_f1" in row]
    if not names:
        raise ValueError(f"{dataset}: no direct-label micro-F1 measurements")
    return sorted(names, key=lambda name: rows[name]["micro_f1"], reverse=True)


def validate(sources: dict) -> None:
    """Check comparable releases and measured values before creating any files."""
    full, sample, bench, ood = (sources[key] for key in DEFAULTS)
    taxonomy = full["taxonomy_sha256"]
    if not taxonomy or sample["taxonomy_sha256"] != taxonomy or ood["taxonomy_sha256"] != taxonomy:
        raise ValueError("Summaries have different taxonomy fingerprints")
    if full["label_count"] != sample["label_count"]:
        raise ValueError("Full and sample summaries have different label counts")
    for summary, datasets in ((full, CONDITIONS), (sample, ("en_test_kev", "tr_test_kev"))):
        reference = set(systems(summary, next(iter(datasets))))
        for dataset in datasets:
            if set(systems(summary, dataset)) != reference:
                raise ValueError(f"{dataset}: inconsistent direct-label systems")
            for name in reference:
                number(summary["metrics"][dataset][name]["micro_f1"], f"{dataset}/{name}")
    if bench["eval_set"] != "en_test":
        raise ValueError("Accuracy/latency comparison requires an English test benchmark")
    if bench["eval_set_sha256"] != full["test_sets"]["en_test"]["sha256"]:
        raise ValueError("Latency benchmark and full metrics use different test inputs")
    if bench["requested_device"] != "cpu":
        raise ValueError("CPU benchmark charts require CPU measurements")
    for name in MEMBERS:
        row, dump = bench["models"][name], full["dumps"][f"{name}/en_test"]
        for key in ("artifact_sha256", "artifact_files_sha256"):
            if not row[key] or row[key] != dump[key]:
                raise ValueError(f"{name}: latency and accuracy use different artifacts")
        if row["device"] not in ("cpu", "cpu+encoder:cpu"):
            raise ValueError(f"{name}: expected CPU timings")
        single = row["single"]
        p50 = number(single["p50_ms"], f"{name}/p50", 1e-12, math.inf)
        p95 = number(single["p95_ms"], f"{name}/p95", p50, math.inf)
        number(single["p99_ms"], f"{name}/p99", p95, math.inf)
        number(row["batched"]["docs_per_second"], f"{name}/throughput", 1e-12, math.inf)
        number(row["artifact_bytes"], f"{name}/artifact_bytes", 1, math.inf)
    for name in ("svm", "hybrid_knn", "transformer"):
        for target in ("tr_test", "cs_sentence_test", "en_test_noisy", "tr_test_noisy"):
            row = full["significance"]["cross_condition"][name][f"en_test->{target}"]["micro_f1"]
            difference = number(row["difference"], "F1 difference", -1, 1)
            low = number(row["ci_low"], "difference CI low", -1, 1)
            number(row["ci_high"], "difference CI high", low, 1)
            measured = (
                full["metrics"]["en_test"][name]["micro_f1"]
                - full["metrics"][target][name]["micro_f1"]
            )
            if not math.isclose(difference, measured, abs_tol=1e-6):
                raise ValueError(f"{name}/{target}: difference does not match recorded F1")
    for name in ("svm", "naive_bayes"):
        for language in ("en", "tr"):
            row = ood["results"][name][language]
            number(row["auroc"], "AUROC")
            low = number(row["auroc_ci_low"], "AUROC CI low")
            number(row["auroc_ci_high"], "AUROC CI high", low)
            number(row["fpr_at_95_tpr"], "FPR at 95% TPR")


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": "#d5dfe5",
            "svg.fonttype": "none",
            "svg.hashsalt": "naltra-release-plots",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def heading(fig, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.025, y=0.975, ha="left", fontsize=19, fontweight="bold")
    fig.text(0.025, 0.915, subtitle, color=MUTED, fontsize=10)


def save(fig, output: Path, name: str, note: str) -> None:
    fig.text(0.025, 0.025, note, fontsize=9, color=MUTED)
    for extension in ("png", "svg"):
        metadata = {"Date": None} if extension == "svg" else {"Software": "NALTRA plot_results"}
        target = output / f"{name}.{extension}"
        fig.savefig(target, dpi=180, metadata=metadata)
        if extension == "svg":
            # Matplotlib leaves trailing spaces in path commands; keep exports Git-clean.
            content = (
                "\n".join(line.rstrip() for line in target.read_text(encoding="utf-8").splitlines())
                + "\n"
            )
            target.write_text(content, encoding="utf-8", newline="\n")
    plt.close(fig)


def bars(summary: dict, datasets: tuple[str, str], title: str, subtitle: str):
    names = systems(summary, datasets[0])
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    fig.subplots_adjust(left=0.24, right=0.96, bottom=0.14, top=0.85)
    y = np.arange(len(names))
    for offset, dataset, color, label in zip(
        (-0.18, 0.18), datasets, (BLUE, ORANGE), ("English", "Turkish"), strict=True
    ):
        values = [summary["metrics"][dataset][name]["micro_f1"] for name in names]
        ax.barh(y + offset, values, height=0.31, color=color, label=label)
        for position, value in zip(y + offset, values, strict=True):
            ax.text(value + 0.012, position, f"{value:.3f}", va="center", fontsize=9)
    ax.set(yticks=y, yticklabels=[LABELS.get(name, name) for name in names], xlim=(0, 1))
    ax.invert_yaxis()
    ax.set_xlabel("Direct-label micro-F1 (higher is better)")
    ax.xaxis.grid(True, alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    heading(fig, title, subtitle)
    return fig


def render(sources: dict, output: Path) -> None:
    full, sample, bench, ood = (sources[key] for key in DEFAULTS)
    style()
    count = full["benchmark"]["label_distribution"]["en_test"]["n_records"]
    sample_count = sample["benchmark"]["label_distribution"]["en_test_kev"]["n_records"]
    save(
        bars(
            full,
            ("en_test", "tr_test"),
            "Full benchmark: English and Turkish",
            f"{count:,} projects per language | {full['label_count']} direct labels",
        ),
        output,
        "full_f1",
        "Archived results | Hierarchy-closed vote omitted: no direct-label F1.",
    )
    save(
        bars(
            sample,
            ("en_test_kev", "tr_test_kev"),
            "Fixed sample: English and Turkish",
            f"{sample_count} matched projects per language | Kev, Jev and Laya are sample-only",
        ),
        output,
        "sample_f1",
        "Sample results must be interpreted separately from the full benchmark.",
    )

    names = systems(full, "en_test")
    matrix = np.array(
        [[full["metrics"][dataset][name]["micro_f1"] for dataset in CONDITIONS] for name in names]
    )
    fig, ax = plt.subplots(figsize=(11, 7.5))
    fig.subplots_adjust(left=0.23, right=0.90, bottom=0.16, top=0.84)
    heat = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    for (row, column), value in np.ndenumerate(matrix):
        ax.text(
            column,
            row,
            f"{value:.3f}",
            ha="center",
            va="center",
            color="white" if value > 0.5 else INK,
        )
    ax.set(
        xticks=np.arange(len(CONDITIONS)),
        xticklabels=list(CONDITIONS.values()),
        yticks=np.arange(len(names)),
        yticklabels=[LABELS[name] for name in names],
    )
    fig.colorbar(heat, ax=ax, fraction=0.04, pad=0.04, label="Micro-F1")
    heading(
        fig,
        "Performance across input conditions",
        "Same systems and project splits | Higher is better",
    )
    save(
        fig,
        output,
        "robustness_heatmap",
        "Code-switch and noisy conditions are synthetic; color scale is 0 to 1.",
    )

    targets = ("tr_test", "cs_sentence_test", "en_test_noisy", "tr_test_noisy")
    fig, ax = plt.subplots(figsize=(10.5, 6))
    fig.subplots_adjust(left=0.20, right=0.95, bottom=0.18, top=0.78)
    for offset, name, color in zip(
        (-0.22, 0, 0.22),
        ("svm", "hybrid_knn", "transformer"),
        (BLUE, ORANGE, "#258674"),
        strict=True,
    ):
        for i, target in enumerate(targets):
            row = full["significance"]["cross_condition"][name][f"en_test->{target}"]["micro_f1"]
            ax.hlines(
                i + offset, 100 * row["ci_low"], 100 * row["ci_high"], color=color, linewidth=2
            )
            ax.scatter(
                100 * row["difference"],
                i + offset,
                color=color,
                s=35,
                label=LABELS[name] if i == 0 else None,
                zorder=3,
            )
    ax.axvline(0, color=MUTED, linestyle="--", linewidth=1)
    ax.set(yticks=np.arange(4), yticklabels=[CONDITIONS[t].replace("\n", " ") for t in targets])
    ax.invert_yaxis()
    ax.set_xlabel("English micro-F1 minus condition micro-F1 (×100; lower is better)")
    ax.xaxis.grid(True, alpha=0.15)
    ax.legend(loc="upper right", bbox_to_anchor=(1, 1.16), frameon=False, ncols=3)
    heading(
        fig,
        "Robustness: change from English",
        "Selected members | Points: F1 loss; lines: paired-bootstrap 95% intervals",
    )
    save(
        fig,
        output,
        "robustness_drops",
        "Absolute F1-point differences, not relative percentages; "
        "intervals cover test sampling only.",
    )

    y = np.arange(len(MEMBERS))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(left=0.15, right=0.95, wspace=0.40, top=0.80, bottom=0.20)
    timing, throughput = axes
    for i, name in enumerate(MEMBERS):
        row = bench["models"][name]
        a, b = row["single"]["p50_ms"], row["single"]["p95_ms"]
        timing.hlines(i, a, b, color=BLUE, linewidth=3)
        timing.scatter(a, i, color=BLUE, label="p50" if i == 0 else None)
        timing.scatter(
            b, i, facecolor="white", edgecolor=BLUE, label="p95" if i == 0 else None, zorder=3
        )
        timing.annotate(
            f"{a:.1f} / {b:.1f}",
            (b, i),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    timing.set_xscale("log")
    timing.set_xlim(1, 3000)
    timing.set_xlabel("Single-text latency, ms (log scale)\nLower is better; labels: p50 / p95")
    timing.legend(loc="upper right", frameon=False)
    values = [bench["models"][name]["batched"]["docs_per_second"] for name in MEMBERS]
    throughput.barh(y, values, color=BLUE, height=0.5)
    for i, value in enumerate(values):
        throughput.text(value + 15, i, f"{value:.1f}", va="center", fontsize=10)
    throughput.set_xlim(0, max(values) * 1.2)
    throughput.set_xlabel(
        f"Batched texts per second (batch size {bench['batch_size']})\nHigher is better"
    )
    for ax in axes:
        ax.set(yticks=y, yticklabels=[LABELS[name] for name in MEMBERS])
        ax.invert_yaxis()
        ax.xaxis.grid(True, alpha=0.15)
        ax.set_axisbelow(True)
    heading(
        fig,
        "CPU inference benchmark",
        f"{bench['samples']} English texts | {bench['warmup']} warm-ups | "
        f"{bench['runtime']['torch_threads']} torch threads",
    )
    save(
        fig,
        output,
        "cpu_benchmark",
        "Recorded CPU run; p50–p95 spans are latency percentiles, not confidence intervals.",
    )

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.18, top=0.84)
    for name, color in zip(MEMBERS, (BLUE, ORANGE, "#258674", "#8058a3", "#b04b5a"), strict=True):
        x = bench["models"][name]["single"]["p50_ms"]
        value = full["metrics"]["en_test"][name]["micro_f1"]
        ax.scatter(x, value, s=90, color=color, zorder=3)
        offset = (-12, -18) if name in ("naive_bayes", "hybrid_knn") else (8, 10)
        ax.annotate(
            LABELS[name], (x, value), xytext=offset, textcoords="offset points", fontsize=11
        )
    ax.set_xscale("log")
    ax.set(
        xlim=(1, 1500),
        ylim=(0, 1),
        xlabel="CPU p50 latency per text, ms (log scale; lower is better)",
        ylabel="English direct-label micro-F1 (higher is better)",
    )
    ax.grid(alpha=0.15)
    heading(
        fig,
        "Accuracy and inference cost",
        "Full English test accuracy and matching measured CPU artifacts",
    )
    save(
        fig,
        output,
        "accuracy_latency",
        "Upper left combines higher accuracy with lower latency. "
        "Ensemble timings were not recorded.",
    )

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    fig.subplots_adjust(left=0.17, right=0.95, bottom=0.22, top=0.80)
    sizes = [bench["models"][name]["artifact_bytes"] / 1e6 for name in MEMBERS]
    ax.barh(y, sizes, color=BLUE, height=0.55)
    for i, value in enumerate(sizes):
        ax.text(value + 15, i, f"{value:,.1f}", va="center", fontsize=10)
    ax.set(
        yticks=y,
        yticklabels=[LABELS[name] for name in MEMBERS],
        xlim=(0, max(sizes) * 1.18),
        xlabel="Recorded artifact size, MB (1 MB = 1,000,000 bytes)",
    )
    ax.invert_yaxis()
    ax.xaxis.grid(True, alpha=0.15)
    ax.set_axisbelow(True)
    heading(
        fig, "Stored model artifacts", "Measured files on disk; external dependencies are excluded"
    )
    save(
        fig,
        output,
        "artifact_size",
        "Hybrid kNN excludes its separately downloaded ~1.1 GB e5 encoder. "
        "Sizes are not total deployment footprints.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.subplots_adjust(left=0.15, right=0.94, wspace=0.45, bottom=0.22, top=0.80)
    for offset, language, color, label in zip(
        (-0.12, 0.12), ("en", "tr"), (BLUE, ORANGE), ("English", "Turkish"), strict=True
    ):
        for i, name in enumerate(("svm", "naive_bayes")):
            row = ood["results"][name][language]
            axes[0].hlines(
                i + offset, row["auroc_ci_low"], row["auroc_ci_high"], color=color, linewidth=2
            )
            axes[0].scatter(row["auroc"], i + offset, color=color, label=label if i == 0 else None)
            axes[1].barh(i + offset, row["fpr_at_95_tpr"], height=0.20, color=color)
            axes[1].text(
                row["fpr_at_95_tpr"] + 0.025,
                i + offset,
                f"{row['fpr_at_95_tpr']:.3f}",
                va="center",
                fontsize=10,
            )
    for ax in axes:
        ax.set(xlim=(0, 1), yticks=(0, 1), yticklabels=("SVM", "Naive Bayes"), ylim=(1.5, -0.5))
        ax.xaxis.grid(True, alpha=0.15)
        ax.set_axisbelow(True)
    axes[0].axvline(0.5, color=MUTED, linestyle="--", linewidth=1)
    axes[0].set_xlabel("AUROC (higher is better)\nLines: bootstrap 95% intervals")
    axes[0].legend(loc="lower left", frameon=False)
    axes[1].set_xlabel("False-positive rate at 95% TPR\nLower is better")
    counts = ood["results"]["svm"]["en"]
    heading(
        fig,
        "Near-OOD detection",
        f"Held-out root: {ood['heldout_root']} | {counts['id_records']:,} in-domain / "
        f"{counts['ood_records']} OOD projects per language",
    )
    save(
        fig,
        output,
        "ood_detection",
        "Separate retrained models; mixed-domain projects excluded. Dashed AUROC = chance level.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key, default in DEFAULTS.items():
        parser.add_argument(f"--{key}", type=Path, default=Path(default))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures/cordis_v1.1.0"))
    args = parser.parse_args(argv)
    sources, provenance = {}, {}
    for key in DEFAULTS:
        path = getattr(args, key)
        raw = path.read_bytes()
        sources[key] = json.loads(raw)
        provenance[key] = {"path": path.as_posix(), "sha256": hashlib.sha256(raw).hexdigest()}
    validate(sources)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render(sources, args.output_dir)
    manifest = {
        "description": "Charts of archived summaries; experiments were not rerun.",
        "sources": provenance,
        "figures": [f"{name}.{ext}" for name in FIGURES for ext in ("png", "svg")],
        "generator": "scripts/plot_results.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(FIGURES)} charts (PNG + SVG) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
