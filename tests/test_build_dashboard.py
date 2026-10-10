"""The interactive results page is built only from validated, summary-bound dumps."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from scripts import build_dashboard
from tests.test_make_figures import benchmark

TIMING = {"single": {"p50_ms": 1.0, "p95_ms": 2.0}, "batched": {"docs_per_second": 9.0}}
LATENCY = {"hardware": {"cpu": "CPU"}, "models": {"svm": TIMING}}
ARTIFACT = ("artifact_sha256", "artifact_files_sha256")
SCORES = {"en": {"id_records": 9, "ood_records": 3, "auroc": 0.8, "fpr_at_95_tpr": 0.4}}
OOD = {"heldout_root": "humanities", "counts": {}, "results": {"svm": SCORES}}


def embedded(html: str) -> dict:
    start = html.index("const DATA = ") + len("const DATA = ")
    return json.loads(html[start : html.index(";\nconst SVGNS")])


def bind_release(tmp_path, args: list[str], latency: dict, ood: dict) -> list[str]:
    """Write latency and OOD summaries recording the scored set, artifacts and taxonomy."""
    summary_path = Path(args[args.index("--summary") + 1])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tested = summary["test_sets"].setdefault("en_test", summary["test_sets"]["test"])
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    latency = {
        "eval_set": "test",
        "eval_set_path": str(tmp_path / "test.jsonl"),
        "eval_set_sha256": tested["sha256"],
    } | latency
    latency["models"] = {
        model: {key: summary["dumps"][f"{model}/test"][key] for key in ARTIFACT} | timing
        for model, timing in latency["models"].items()
    }
    inputs = {"en/train": "0" * 64, "en/test": tested["sha256"]}
    ood = {"taxonomy_sha256": summary["taxonomy_sha256"], "inputs": inputs} | ood
    for name, record in (("latency", latency), ("ood", ood)):
        (tmp_path / f"{name}.json").write_text(json.dumps(record), encoding="utf-8")
    return [*args, "--latency", str(tmp_path / "latency.json"), "--ood", str(tmp_path / "ood.json")]


SAMPLE_SETS = ("en_test_kev", "tr_test_kev")
SAMPLE_DROP = {"difference": 0.04, "ci_low": 0.02, "ci_high": 0.07}
SAMPLE_METRICS = {
    "svm": {"micro_f1": 0.55, "macro_f1": 0.25, "hmicro_f1": 0.66},
    "kev": {"micro_f1": 0.27, "macro_f1": 0.16, "hmicro_f1": 0.4},
    "jev": {"micro_f1": 0.31, "macro_f1": 0.2, "hmicro_f1": 0.45},
    "hard_closed": {"hmicro_f1": 0.61, "hierarchy_violation_rate": 0.0},
}


def with_sample(tmp_path, args: list[str], edit=lambda sample: sample) -> list[str]:
    """Write a Kev/Jev sample summary on the main summary's taxonomy, with local paths."""
    summary = json.loads(Path(args[args.index("--summary") + 1]).read_text(encoding="utf-8"))
    en, tr = SAMPLE_SETS
    drop = {"micro_f1": SAMPLE_DROP | {"p_holm": 0.001}}
    sample = {
        "taxonomy_sha256": summary["taxonomy_sha256"],
        "members": ["svm", "kev", "jev"],
        "test_sets": {
            name: {"path": str(tmp_path / f"{name}.jsonl"), "sha256": str(index) * 64}
            for index, name in enumerate(SAMPLE_SETS)
        },
        "metrics": {name: SAMPLE_METRICS for name in SAMPLE_SETS},
        "significance": {
            "reference_set": en,
            "cross_condition": {s: {f"{en}->{tr}": drop} for s in ("svm", "kev", "jev")},
        },
        "benchmark": {
            "systems": {"jev": {"role": "new", "model": str(tmp_path)}},
            "label_distribution": {en: {"n_records": 120}},
        },
    }
    (tmp_path / "sample.json").write_text(json.dumps(edit(sample)), encoding="utf-8")
    return [*args, "--sample-summary", str(tmp_path / "sample.json")]


def dashboard_args(tmp_path) -> list[str]:
    figure_args = benchmark(tmp_path)
    summary = figure_args[figure_args.index("--summary") + 1]
    test_sets = figure_args[figure_args.index("--test-sets") + 1]
    return [
        "--summary",
        summary,
        "--predictions",
        str(tmp_path / "predictions"),
        "--test-sets",
        test_sets,
        "--output",
        str(tmp_path / "site/index.html"),
        "--fragment",
        str(tmp_path / "site/fragment.html"),
    ]


def test_page_requests_no_external_assets(tmp_path) -> None:
    assert build_dashboard.main(dashboard_args(tmp_path)) == 0
    page = (tmp_path / "site/index.html").read_text(encoding="utf-8")
    external = r"(?:src|href)\s*=\s*[\"']?(?:https?:)?//|url\(\s*[\"']?(?:https?:)?//|@import"
    assert not re.findall(external, page, re.I)


def test_fewer_than_three_voters_are_noted_as_a_degenerate_vote(tmp_path) -> None:
    assert build_dashboard.main(dashboard_args(tmp_path)) == 0
    meta = embedded((tmp_path / "site/index.html").read_text(encoding="utf-8"))["meta"]
    assert len(meta["members"]) == 2 and "degenerate" in meta["pending"]
    assert not any(word in meta["pending"] for word in ("GPU", "five-family", "planned"))


def test_page_embeds_metrics_and_per_system_detail(tmp_path) -> None:
    assert build_dashboard.main(dashboard_args(tmp_path)) == 0
    page = (tmp_path / "site/index.html").read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>") and "<title>NALTRA Results</title>" in page
    data = embedded(page)
    assert data["meta"]["test_sets"] == ["test"]
    assert set(data["metrics"]["test"]) >= {"svm", "naive_bayes", "hard_majority", "soft"}
    detail = data["detail"]["test"]
    assert set(detail) >= {"svm", "naive_bayes", "hard_majority", "weighted_soft"}
    confusion = detail["svm"]["confusion"]
    assert len(confusion["matrix"]) == len(confusion["roots"]) == len(confusion["support"])
    assert detail["svm"]["reliability"]["count"] and "reliability" not in detail["hard_majority"]
    assert all(0 <= v <= 1 for v in detail["svm"]["depth"].values())
    assert data["sample"] is None
    fragment = (tmp_path / "site/fragment.html").read_text(encoding="utf-8")
    assert not fragment.lstrip().lower().startswith("<!doctype") and "const DATA = " in fragment


def test_embedded_json_cannot_close_the_script_tag(tmp_path) -> None:
    args = dashboard_args(tmp_path)
    summary_path = args[args.index("--summary") + 1]
    summary = json.loads(open(summary_path, encoding="utf-8").read())
    summary["metrics"]["test"]["svm"]["note"] = "</script><script>alert(1)</script>"
    open(summary_path, "w", encoding="utf-8").write(json.dumps(summary))
    assert build_dashboard.main(args) == 0
    page = (tmp_path / "site/index.html").read_text(encoding="utf-8")
    assert "</script><script>alert(1)" not in page


def test_page_carries_the_summary_benchmark(tmp_path) -> None:
    args = dashboard_args(tmp_path)
    summary_path = args[args.index("--summary") + 1]
    summary = json.loads(open(summary_path, encoding="utf-8").read())
    summary["benchmark"] = {
        "primary_metric": "micro_f1",
        "systems": {"svm": {"role": "baseline", "architecture": "Linear SVM", "model": "SVC"}},
        "label_distribution": {"test": {"n_records": 4, "n_labels": 2}},
    }
    page = tmp_path / "site/index.html"
    open(summary_path, "w", encoding="utf-8").write(json.dumps(summary))
    assert build_dashboard.main(args) == 0
    assert embedded(page.read_text(encoding="utf-8"))["benchmark"] == summary["benchmark"]
    del summary["benchmark"]
    open(summary_path, "w", encoding="utf-8").write(json.dumps(summary))
    assert build_dashboard.main(args) == 0
    assert embedded(page.read_text(encoding="utf-8"))["benchmark"] is None


@pytest.mark.parametrize(
    "edit",
    [
        lambda s: s["metrics"]["test"]["svm"].update(micro_f1=7.5),
        lambda s: s["metrics"]["test"]["svm"].update(micro_f1=-0.1),
        lambda s: s["metrics"]["test"]["svm"].update(micro_f1=10**400),
        lambda s: s.update(benchmark={"primary_metric": ["micro_f1"]}),
        lambda s: s.update(benchmark={"secondary_metrics": [{"a": 1}]}),
        lambda s: s.update(benchmark={"secondary_metrics": "macro_f1"}),
    ],
    ids=[
        "above one",
        "negative",
        "huge integer",
        "listed primary",
        "unhashable secondary",
        "text secondary",
    ],
)
def test_malformed_summary_metrics_build_nothing(tmp_path, capsys, edit) -> None:
    args = dashboard_args(tmp_path)
    summary_path = Path(args[args.index("--summary") + 1])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    edit(summary)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    capsys.readouterr()
    assert build_dashboard.main(args) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    assert not (tmp_path / "site/index.html").exists()


def test_tampered_dump_builds_nothing(tmp_path) -> None:
    args = dashboard_args(tmp_path)
    dump = tmp_path / "predictions/svm/test.npz"
    with np.load(dump) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["scores"] = arrays["scores"][:, ::-1].copy()
    np.savez_compressed(dump, **arrays)
    assert build_dashboard.main(args) == 1
    assert not (tmp_path / "site/index.html").exists()


def test_page_carries_release_bound_latency_and_ood_without_local_paths(tmp_path) -> None:
    args = bind_release(tmp_path, dashboard_args(tmp_path), LATENCY, OOD)
    assert build_dashboard.main(with_sample(tmp_path, args)) == 0
    page = (tmp_path / "site/index.html").read_text(encoding="utf-8")
    data = embedded(page)
    assert data["latency"]["models"]["svm"].items() >= TIMING.items() and data["ood"]["results"]
    assert data["sample"]["en"]["jev"]
    assert str(tmp_path) not in page and json.dumps(str(tmp_path))[1:-1] not in page


def test_page_carries_the_sample_table_of_every_system(tmp_path) -> None:
    args = dashboard_args(tmp_path)
    summary_path = Path(args[args.index("--summary") + 1])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["benchmark"] = {"primary_metric": "micro_f1", "secondary_metrics": ["hmicro_f1"]}
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    assert build_dashboard.main(with_sample(tmp_path, args)) == 0
    sample = embedded((tmp_path / "site/index.html").read_text(encoding="utf-8"))["sample"]
    shown = {
        system: {key: scores[key] for key in ("micro_f1", "hmicro_f1") if key in scores}
        for system, scores in SAMPLE_METRICS.items()
    }
    assert sample == {
        "records": 120,
        "members": ["svm", "kev", "jev"],
        "headline": ["micro_f1", "hmicro_f1"],
        "en": shown,
        "tr": shown,
        "drop": {system: SAMPLE_DROP for system in ("svm", "kev", "jev")},
        "roles": {"jev": "new"},
    }


def renamed(sample: dict, names: tuple[str, str]) -> dict:
    """The same sample with its two test sets renamed."""
    sets = dict(zip(SAMPLE_SETS, names, strict=True))
    return sample | {
        "test_sets": {sets[k]: v for k, v in sample["test_sets"].items()},
        "metrics": {sets[k]: v for k, v in sample["metrics"].items()},
    }


def rescore(sample: dict, name: str, system: str, scores: object) -> dict:
    return sample | {"metrics": sample["metrics"] | {name: SAMPLE_METRICS | {system: scores}}}


def redropped(sample: dict, drop: dict) -> dict:
    cross = {"kev": {"->".join(SAMPLE_SETS): {"micro_f1": drop}}}
    return sample | {"significance": {"cross_condition": cross}}


def rehashed(sample: dict, turkish: dict) -> dict:
    """The same sample with another Turkish test set entry."""
    return sample | {"test_sets": sample["test_sets"] | {"tr_test_kev": turkish}}


def recounted(sample: dict, records: object) -> dict:
    counts = {"label_distribution": {"en_test_kev": {"n_records": records}}}
    return sample | {"benchmark": counts}


MALFORMED_SAMPLE = {
    "a list": lambda s: [s],
    "null": lambda s: None,
    "another taxonomy": lambda s: s | {"taxonomy_sha256": "0" * 64},
    "no Turkish set": lambda s: s | {"test_sets": {"en_test_kev": s["test_sets"]["en_test_kev"]}},
    "a third set": lambda s: s | {"test_sets": s["test_sets"] | {"cs_chunk_test": {}}},
    "validation sets": lambda s: renamed(s, ("en_val_kev", "tr_val_kev")),
    "two English sets": lambda s: renamed(s, ("en_test_kev", "en_test_noisy")),
    "no Turkish metrics": lambda s: s | {"metrics": {"en_test_kev": SAMPLE_METRICS}},
    "one file for both languages": lambda s: rehashed(s, s["test_sets"]["en_test_kev"]),
    "an unhashed Turkish set": lambda s: rehashed(s, {"path": "tr_test_kev.jsonl"}),
    "a text metric": lambda s: rescore(s, "tr_test_kev", "jev", {"micro_f1": "0.31"}),
    "an infinite metric": lambda s: rescore(s, "en_test_kev", "kev", {"micro_f1": float("inf")}),
    "a huge integer metric": lambda s: rescore(s, "en_test_kev", "kev", {"micro_f1": 10**400}),
    "a metric above one": lambda s: rescore(s, "en_test_kev", "kev", {"micro_f1": 7.5}),
    "a negative metric": lambda s: rescore(s, "tr_test_kev", "jev", {"micro_f1": -0.31}),
    "a negative record count": lambda s: recounted(s, -120),
    "a text record count": lambda s: recounted(s, "120"),
    "a null system": lambda s: rescore(s, "en_test_kev", "kev", None),
    "a system scored in one language": lambda s: rescore(s, "en_test_kev", "extra", {}),
    "a scoreless system": lambda s: rescore(
        rescore(s, "en_test_kev", "kev", {}), "tr_test_kev", "kev", {}
    ),
    "a text drop": lambda s: redropped(s, SAMPLE_DROP | {"difference": "0.04"}),
    "a drop without its interval": lambda s: redropped(s, {"difference": 0.04}),
    "text members": lambda s: s | {"members": "svm kev jev"},
}


@pytest.mark.parametrize("edit", MALFORMED_SAMPLE.values(), ids=MALFORMED_SAMPLE)
def test_malformed_sample_summary_builds_nothing(tmp_path, capsys, edit) -> None:
    args = with_sample(tmp_path, dashboard_args(tmp_path), edit)
    capsys.readouterr()
    assert build_dashboard.main(args) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    assert not (tmp_path / "site/index.html").exists()


STALE = {
    "other latency set": ("latency", lambda r: r | {"eval_set_sha256": "0" * 64}),
    "unhashed latency": ("latency", lambda r: r | {"eval_set_sha256": None}),
    "unscored latency set": ("latency", lambda r: r | {"eval_set": "tr_test"}),
    "other OOD test": ("ood", lambda r: r | {"inputs": r["inputs"] | {"en/test": "0" * 64}}),
    "unscored OOD test": ("ood", lambda r: r | {"inputs": r["inputs"] | {"tr/test": "0" * 64}}),
    "no OOD test": ("ood", lambda r: r | {"inputs": {"en/train": "0" * 64}}),
    "OOD on another taxonomy": ("ood", lambda r: r | {"taxonomy_sha256": "0" * 64}),
    "latency of older artifacts": ("latency", lambda r: retimed(r, "artifact_sha256")),
    "latency of older artifact files": ("latency", lambda r: retimed(r, "artifact_files_sha256")),
    "latency of an unscored model": ("latency", lambda r: r | {"models": {"bilstm": TIMING}}),
    "null OOD inputs": ("ood", lambda r: r | {"inputs": None}),
    "listed latency models": ("latency", lambda r: r | {"models": list(r["models"].values())}),
    "listed latency set": ("latency", lambda r: r | {"eval_set": [r["eval_set"]]}),
    "null OOD results": ("ood", lambda r: r | {"results": None}),
    "listed OOD results": ("ood", lambda r: r | {"results": list(r["results"].values())}),
    "null OOD model result": ("ood", lambda r: r | {"results": {"svm": None}}),
    "listed OOD model result": ("ood", lambda r: r | {"results": {"svm": [SCORES]}}),
    "empty OOD model result": ("ood", lambda r: r | {"results": {"svm": {}}}),
    "listed OOD language result": ("ood", lambda r: r | {"results": {"svm": {"en": [0.8]}}}),
    "OOD without counts": ("ood", lambda r: r | {"counts": None}),
    "OOD language without a test input": (
        "ood",
        lambda r: rescored(r, SCORES | {"tr": SCORES["en"]}),
    ),
    "OOD result for another language": ("ood", lambda r: rescored(r, {"tr": SCORES["en"]})),
    "OOD result without auroc": (
        "ood",
        lambda r: rescored(r, {"en": without(SCORES["en"], "auroc")}),
    ),
    "OOD result without FPR": (
        "ood",
        lambda r: rescored(r, {"en": without(SCORES["en"], "fpr_at_95_tpr")}),
    ),
    "OOD result without counts": (
        "ood",
        lambda r: rescored(r, {"en": without(SCORES["en"], "id_records")}),
    ),
    "OOD result with a text auroc": (
        "ood",
        lambda r: rescored(r, {"en": SCORES["en"] | {"auroc": "0.8"}}),
    ),
    "OOD result with a text interval": (
        "ood",
        lambda r: rescored(r, {"en": SCORES["en"] | {"auroc_ci_low": "x"}}),
    ),
    "OOD auroc above one": ("ood", lambda r: rescored(r, {"en": SCORES["en"] | {"auroc": 7.5}})),
    "OOD negative FPR": (
        "ood",
        lambda r: rescored(r, {"en": SCORES["en"] | {"fpr_at_95_tpr": -0.4}}),
    ),
    "OOD interval above one": (
        "ood",
        lambda r: rescored(r, {"en": SCORES["en"] | {"auroc_ci_high": 1.2}}),
    ),
    "OOD negative count": (
        "ood",
        lambda r: rescored(r, {"en": SCORES["en"] | {"ood_records": -3}}),
    ),
    "empty OOD results": ("ood", lambda r: r | {"results": {}}),
    "null OOD summary": ("ood", lambda r: None),
    "latency without single timing": ("latency", lambda r: retimed(r, "single", None)),
    "latency without batched timing": ("latency", lambda r: retimed(r, "batched", None)),
    "latency without p95": ("latency", lambda r: retimed(r, "single", {"p50_ms": 1.0})),
    "latency with a text rate": (
        "latency",
        lambda r: retimed(r, "batched", {"docs_per_second": "9"}),
    ),
    "latency with a text size": ("latency", lambda r: retimed(r, "artifact_bytes", "big")),
    "latency with a negative p50": (
        "latency",
        lambda r: retimed(r, "single", {"p50_ms": -1.0, "p95_ms": 2.0}),
    ),
    "latency with a negative rate": (
        "latency",
        lambda r: retimed(r, "batched", {"docs_per_second": -9.0}),
    ),
    "latency with a negative size": ("latency", lambda r: retimed(r, "artifact_bytes", -1)),
    "latency with a negative sample count": ("latency", lambda r: r | {"samples": -200}),
    "latency without hardware": ("latency", lambda r: without(r, "hardware")),
    "latency without models": ("latency", lambda r: r | {"models": {}}),
    "null latency summary": ("latency", lambda r: None),
}


def without(record: dict, field: str) -> dict:
    return {key: value for key, value in record.items() if key != field}


def rescored(ood: dict, scores: dict) -> dict:
    return ood | {"results": {"svm": scores}}


def retimed(latency: dict, field: str, value: object = "0" * 64) -> dict:
    """The same timings with one field changed, as a stale or partial run leaves them."""
    models = {model: timing | {field: value} for model, timing in latency["models"].items()}
    if value is None:
        models = {model: without(timing, field) for model, timing in models.items()}
    return latency | {"models": models}


@pytest.mark.parametrize("name, edit", STALE.values(), ids=STALE)
def test_latency_or_ood_from_another_release_builds_nothing(tmp_path, capsys, name, edit) -> None:
    args = bind_release(tmp_path, dashboard_args(tmp_path), LATENCY, OOD)
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(edit(json.loads(path.read_text(encoding="utf-8")))), "utf-8")
    capsys.readouterr()
    assert build_dashboard.main(args) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    assert not (tmp_path / "site/index.html").exists()


@pytest.mark.parametrize("benchmark", [["micro_f1"], "micro_f1", 7], ids=["list", "text", "number"])
def test_summary_benchmark_that_is_not_an_object_builds_nothing(
    tmp_path, capsys, benchmark
) -> None:
    args = dashboard_args(tmp_path)
    summary_path = Path(args[args.index("--summary") + 1])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["benchmark"] = benchmark
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    for run in (args, with_sample(tmp_path, args)):
        capsys.readouterr()
        assert build_dashboard.main(run) == 1
        assert json.loads(capsys.readouterr().out)["status"] == "failed"
    assert not (tmp_path / "site/index.html").exists()


@pytest.mark.parametrize("entries", ["sha", None, ["sha"]], ids=["text", "null", "list"])
def test_malformed_test_set_entry_builds_nothing(tmp_path, capsys, entries) -> None:
    args = dashboard_args(tmp_path)
    summary_path = Path(args[args.index("--summary") + 1])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["test_sets"]["test"] = entries
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    capsys.readouterr()
    assert build_dashboard.main(args) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    assert not (tmp_path / "site/index.html").exists()


def test_script_starts_when_run_directly() -> None:
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/build_dashboard.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=root.parent,
    )
    assert result.returncode == 0, result.stderr
