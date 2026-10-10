"""Offline ensemble benchmark over prediction dumps: thresholds, voting, metrics."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scripts import evaluate_ensemble
from tests.test_provenance import absolute_paths

from naltra.data.loader import save_jsonl
from naltra.data.manifest import compute_file_sha256, get_environment_metadata
from naltra.evaluation.bootstrap import SEED
from naltra.utils import provenance
from naltra.utils.config import load_yaml

LABELS = ["acoustics", "optics"]


def write_set(tmp_path, name, count, rng, pair_prefix=None):
    rows = []
    for index in range(count):
        target = [LABELS[index % 2]] if index % 3 else LABELS
        rows.append(
            {
                "id": f"{name}:{index}",
                "pair_id": f"cordis:{pair_prefix or name}:{index}",
                "text": f"{name} text {index}",
                "labels_direct": target,
                "labels": [*target, "physical_sciences", "natural_sciences"],
            }
        )
    save_jsonl(rows, tmp_path / f"{name}.jsonl")
    return rows


def write_dump(directory, family, name, rows, noise, rng, scores=None):
    truth = np.array([[label in r["labels_direct"] for label in LABELS] for r in rows], float)
    if scores is None:
        scores = np.clip(truth * 0.7 + rng.random(truth.shape) * noise, 0, 1).astype(np.float32)
    target = directory / family
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target / f"{name}.npz",
        ids=np.array([r["id"] for r in rows]),
        labels=np.array(LABELS),
        scores=scores,
    )
    sidecar = {
        "model": family,
        "eval_set": name,
        "input_sha256": compute_file_sha256(directory.parent / f"{name}.jsonl"),
        "records": len(rows),
        "artifact_sha256": "a" * 64,
        "artifact_files_sha256": {"model.joblib": "b" * 64},
        "config_sha256": "e" * 64,
    }
    (target / f"{name}.json").write_text(json.dumps(sidecar), encoding="utf-8")


def test_benchmark_tunes_on_validation_and_scores_members_and_ensembles(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    validation = write_set(tmp_path, "val", 60, rng)
    test = write_set(tmp_path, "test", 30, rng)
    predictions = tmp_path / "predictions"
    for family, noise in (
        ("svm", 0.4),
        ("bilstm", 0.5),
        ("transformer", 0.3),
        ("naive_bayes", 0.9),
    ):
        write_dump(predictions, family, "val", validation, noise, rng)
        write_dump(predictions, family, "test", test, noise, rng)
    output = tmp_path / "metrics"
    code = evaluate_ensemble.main(
        [
            "--predictions",
            str(predictions),
            "--members",
            "svm",
            "bilstm",
            "transformer",
            "--baselines",
            "naive_bayes",
            "--tune-sets",
            f"val={tmp_path / 'val.jsonl'}",
            "--test-sets",
            f"test={tmp_path / 'test.jsonl'}",
            "--output-dir",
            str(output),
        ]
    )
    assert code == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["thresholds"]) == {"svm", "bilstm", "transformer", "naive_bayes"}
    systems = summary["metrics"]["test"]
    for system in (
        "svm",
        "naive_bayes",
        "hard_majority",
        "hard_k",
        "hard_closed",
        "soft",
        "weighted_soft",
    ):
        assert 0.0 <= systems[system]["hmicro_f1"] <= 1.0
    assert "micro_f1" not in systems["hard_closed"]
    assert systems["hard_closed"]["hierarchy_violation_rate"] == 0.0
    assert "ece" in systems["soft"] and "ece" not in systems["hard_majority"]
    assert set(summary["leave_one_out"]["test"]) == {"svm", "bilstm", "transformer"}
    assert "| system |" in (output / "summary.md").read_text(encoding="utf-8")


def test_misaligned_member_dumps_are_rejected(tmp_path):
    rng = np.random.default_rng(1)
    rows = write_set(tmp_path, "val", 10, rng)
    write_set(tmp_path, "test", 10, rng)
    predictions = tmp_path / "predictions"
    write_dump(predictions, "svm", "val", rows, 0.3, rng)
    write_dump(predictions, "bilstm", "val", list(reversed(rows)), 0.3, rng)
    code = evaluate_ensemble.main(
        [
            "--predictions",
            str(predictions),
            "--members",
            "svm",
            "bilstm",
            "--tune-sets",
            f"val={tmp_path / 'val.jsonl'}",
            "--test-sets",
            f"test={tmp_path / 'test.jsonl'}",
            "--output-dir",
            str(tmp_path / "metrics"),
        ]
    )
    assert code == 1


def benchmark_args(tmp_path, members, tune="val", test="test"):
    return [
        "--predictions",
        str(tmp_path / "predictions"),
        "--members",
        *members,
        "--tune-sets",
        f"{tune}={tmp_path / f'{tune}.jsonl'}",
        "--test-sets",
        f"{test}={tmp_path / f'{test}.jsonl'}",
        "--output-dir",
        str(tmp_path / "metrics"),
    ]


def test_duplicate_members_are_rejected(tmp_path):
    with pytest.raises(SystemExit) as result:
        evaluate_ensemble.main(benchmark_args(tmp_path, ["svm", "bilstm", "svm"]))
    assert result.value.code == 2


def test_test_split_cannot_be_used_for_tuning(tmp_path):
    with pytest.raises(SystemExit) as result:
        evaluate_ensemble.main(
            benchmark_args(tmp_path, ["svm", "bilstm"], tune="test", test="test")
        )
    assert result.value.code == 2


def test_summary_records_the_tuning_protocol(tmp_path):
    rng = np.random.default_rng(2)
    validation = write_set(tmp_path, "val", 40, rng)
    test = write_set(tmp_path, "test", 20, rng)
    for family in ("svm", "bilstm"):
        write_dump(tmp_path / "predictions", family, "val", validation, 0.3, rng)
        write_dump(tmp_path / "predictions", family, "test", test, 0.3, rng)
    assert evaluate_ensemble.main(benchmark_args(tmp_path, ["svm", "bilstm"])) == 0
    summary = json.loads((tmp_path / "metrics/summary.json").read_text(encoding="utf-8"))
    assert set(summary["tune_sets"]) == {"val"} and set(summary["test_sets"]) == {"test"}
    assert len(summary["tune_sets"]["val"]["sha256"]) == 64
    assert summary["threshold_grid"][0] == pytest.approx(0.02)


def write_benchmark(tmp_path, seed=3):
    rng = np.random.default_rng(seed)
    sets = {name: write_set(tmp_path, name, 20, rng) for name in ("val", "test")}
    for family in ("svm", "bilstm"):
        for name, rows in sets.items():
            write_dump(tmp_path / "predictions", family, name, rows, 0.3, rng)
    return sets


def last_error(capsys):
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])["error"]


def test_dumps_of_an_edited_gold_set_are_rejected_despite_matching_ids(tmp_path, capsys):
    rows = write_benchmark(tmp_path)["val"]
    edited = [r | {"text": f"edited {r['text']}", "labels_direct": ["optics"]} for r in rows]
    save_jsonl(edited, tmp_path / "val.jsonl")
    assert evaluate_ensemble.main(benchmark_args(tmp_path, ["svm", "bilstm"])) == 1
    assert last_error(capsys).startswith("svm/val:")


@pytest.mark.parametrize("field, value", [("model", "nb"), ("eval_set", "test"), ("records", 19)])
def test_dump_sidecar_must_describe_the_evaluated_set(tmp_path, capsys, field, value):
    write_benchmark(tmp_path)
    sidecar = tmp_path / "predictions/svm/val.json"
    patched = json.loads(sidecar.read_text(encoding="utf-8")) | {field: value}
    sidecar.write_text(json.dumps(patched), encoding="utf-8")
    assert evaluate_ensemble.main(benchmark_args(tmp_path, ["svm", "bilstm"])) == 1
    assert last_error(capsys).startswith("svm/val:")


def test_dump_without_sidecar_is_rejected(tmp_path):
    write_benchmark(tmp_path)
    (tmp_path / "predictions/bilstm/test.json").unlink()
    assert evaluate_ensemble.main(benchmark_args(tmp_path, ["svm", "bilstm"])) == 1


def test_summary_records_dump_provenance(tmp_path):
    write_benchmark(tmp_path)
    assert evaluate_ensemble.main(benchmark_args(tmp_path, ["svm", "bilstm"])) == 0
    summary = json.loads((tmp_path / "metrics/summary.json").read_text(encoding="utf-8"))
    dumps = summary["dumps"]
    assert set(dumps) == {"svm/val", "svm/test", "bilstm/val", "bilstm/test"}
    assert dumps["svm/val"] == {
        "npz_sha256": compute_file_sha256(tmp_path / "predictions/svm/val.npz"),
        "input_sha256": compute_file_sha256(tmp_path / "val.jsonl"),
        "artifact_sha256": "a" * 64,
        "artifact_files_sha256": {"model.joblib": "b" * 64},
        "config_sha256": "e" * 64,
    }


def test_summary_records_code_environment_and_seed_without_local_paths(tmp_path):
    write_benchmark(tmp_path)
    args = benchmark_args(tmp_path, ["svm", "bilstm"]) + ["--bootstrap", "50"]
    assert evaluate_ensemble.main(args) == 0
    summary = json.loads((tmp_path / "metrics/summary.json").read_text(encoding="utf-8"))
    assert summary["code_commit"] == provenance.commit()
    assert summary["environment"] == get_environment_metadata()
    assert summary["bootstrap_seed"] == SEED
    digest = compute_file_sha256(tmp_path / "test.jsonl")
    assert summary["test_sets"] == {"test": {"path": "test.jsonl", "sha256": digest}}
    assert absolute_paths(summary, tmp_path) == []
    inside = provenance.REPO_ROOT / "taxonomy" / "taxonomy.json"
    assert evaluate_ensemble.provenance([("taxonomy", inside)])["taxonomy"]["path"] == (
        "taxonomy/taxonomy.json"
    )


def test_summary_records_no_seed_when_the_bootstrap_is_off(tmp_path):
    write_benchmark(tmp_path)
    args = benchmark_args(tmp_path, ["svm", "bilstm"]) + ["--bootstrap", "0"]
    assert evaluate_ensemble.main(args) == 0
    summary = json.loads((tmp_path / "metrics/summary.json").read_text(encoding="utf-8"))
    assert summary["bootstrap_seed"] is None and "significance" not in summary


@pytest.mark.parametrize(
    "field, value",
    [
        ("artifact_sha256", "c" * 64),
        ("artifact_files_sha256", {"model.joblib": "d" * 64}),
        ("config_sha256", "f" * 64),
    ],
)
def test_family_dumps_must_come_from_one_artifact_and_config(tmp_path, capsys, field, value):
    write_benchmark(tmp_path)
    sidecar = tmp_path / "predictions/svm/test.json"
    patched = json.loads(sidecar.read_text(encoding="utf-8")) | {field: value}
    sidecar.write_text(json.dumps(patched), encoding="utf-8")
    assert evaluate_ensemble.main(benchmark_args(tmp_path, ["svm", "bilstm"])) == 1
    assert last_error(capsys).startswith("svm: dumps come from different artifacts/configs")
    assert not (tmp_path / "metrics/summary.json").exists()


def test_family_identity_treats_missing_artifact_fields_as_null():
    service = {"artifact_sha256": None, "artifact_files_sha256": None, "config_sha256": "e" * 64}
    evaluate_ensemble.check_family_identity(
        {"kev/val": {"config_sha256": "e" * 64}, "kev/test": service, "svm/val": {}}
    )
    with pytest.raises(ValueError, match="^kev: dumps come from different artifacts/configs"):
        evaluate_ensemble.check_family_identity(
            {"kev/val": service, "kev/test": service | {"config_sha256": None}}
        )


@pytest.mark.parametrize(
    "tune_patch, test_patch",
    [
        ({"split": "test"}, {}),
        ({"project_id": "cordis-7"}, {"project_id": "cordis-7"}),
        ({}, {"pair_id": "cordis:val:0"}),
        ({}, {"text": "  VAL Text   0 "}),
    ],
)
def test_tuning_and_test_sets_must_not_share_projects(tmp_path, tune_patch, test_patch):
    sets = write_benchmark(tmp_path)
    for name, patch in (("val", tune_patch), ("test", test_patch)):
        rows = sets[name]
        save_jsonl([rows[0] | patch, *rows[1:]], tmp_path / f"{name}.jsonl")
    with pytest.raises(SystemExit) as result:
        evaluate_ensemble.main(benchmark_args(tmp_path, ["svm", "bilstm"]))
    assert result.value.code == 2


def test_language_siblings_of_one_project_may_share_the_tuning_side(tmp_path):
    rng = np.random.default_rng(4)
    english = write_set(tmp_path, "val", 20, rng)
    turkish = [r | {"id": f"tr:{r['id']}", "text": f"metin {i}"} for i, r in enumerate(english)]
    save_jsonl(turkish, tmp_path / "val_tr.jsonl")
    test = write_set(tmp_path, "test", 20, rng)
    for family in ("svm", "bilstm"):
        for name, rows in (("val", english), ("val_tr", turkish), ("test", test)):
            write_dump(tmp_path / "predictions", family, name, rows, 0.3, rng)
    args = benchmark_args(tmp_path, ["svm", "bilstm"])
    args.insert(args.index("--test-sets"), f"val_tr={tmp_path / 'val_tr.jsonl'}")
    assert evaluate_ensemble.main(args) == 0


@pytest.mark.parametrize(
    "labels, transform",
    [
        (LABELS, lambda s: s + 1.0),
        (LABELS, lambda s: s - 1.0),
        (LABELS, lambda s: np.where(s > 0.5, np.nan, s)),
        (LABELS, lambda s: np.where(s > 0.5, np.inf, s)),
        (LABELS, lambda s: s[:, :1]),
        (LABELS, lambda s: s.ravel()),
        (["optics", "optics"], lambda s: s),
        ([1, 2], lambda s: s),
        ([], lambda s: s[:, :0]),
    ],
)
def test_invalid_score_dumps_are_rejected(tmp_path, capsys, labels, transform):
    write_benchmark(tmp_path)
    path = tmp_path / "predictions/svm/val.npz"
    with np.load(path) as loaded:
        dump = dict(loaded)
    np.savez_compressed(
        path, ids=dump["ids"], labels=np.array(labels), scores=transform(dump["scores"])
    )
    assert evaluate_ensemble.main(benchmark_args(tmp_path, ["svm", "bilstm"])) == 1
    assert last_error(capsys).startswith("svm/val:")


def test_summary_reports_bootstrap_significance(tmp_path):
    rng = np.random.default_rng(3)
    validation = write_set(tmp_path, "val", 60, rng)
    english = write_set(tmp_path, "test_en", 40, rng, pair_prefix="test")
    turkish = write_set(tmp_path, "test_tr", 40, rng, pair_prefix="test")
    for family, noise in (("svm", 0.3), ("bilstm", 0.5), ("transformer", 0.4)):
        for name, rows in (("val", validation), ("test_en", english), ("test_tr", turkish)):
            write_dump(tmp_path / "predictions", family, name, rows, noise, rng)
    code = evaluate_ensemble.main(
        [
            "--predictions",
            str(tmp_path / "predictions"),
            "--members",
            "svm",
            "bilstm",
            "transformer",
            "--tune-sets",
            f"val={tmp_path / 'val.jsonl'}",
            "--test-sets",
            f"test_en={tmp_path / 'test_en.jsonl'}",
            f"test_tr={tmp_path / 'test_tr.jsonl'}",
            "--bootstrap",
            "200",
            "--output-dir",
            str(tmp_path / "metrics"),
        ]
    )
    assert code == 0
    summary = json.loads((tmp_path / "metrics/summary.json").read_text(encoding="utf-8"))
    significance = summary["significance"]
    assert significance["resamples"] == 200
    assert significance["best_member"] in {"svm", "bilstm", "transformer"}
    versus = significance["vs_best_member"]["test_en"]
    assert set(versus) == {"hard_majority", "hard_k", "soft", "weighted_soft"}
    result = versus["hard_majority"]["micro_f1"]
    assert {"difference", "ci_low", "ci_high", "p_value", "p_holm"} <= set(result)
    assert result["p_holm"] >= result["p_value"]
    cross = significance["cross_condition"]["svm"]["test_en->test_tr"]["macro_f1"]
    assert {"difference", "ci_low", "ci_high", "p_holm"} <= set(cross)
    assert "## Significance" in (tmp_path / "metrics/summary.md").read_text(encoding="utf-8")


def test_duplicated_project_variants_do_not_narrow_ensemble_intervals(tmp_path):
    rng = np.random.default_rng(5)
    validation = write_set(tmp_path, "val", 40, rng)
    single = write_set(tmp_path, "test", 20, rng)
    doubled = [
        variant
        for r in single
        for variant in (r | {"id": f"en:{r['id']}"}, r | {"id": f"tr:{r['id']}", "text": "x"})
    ]
    save_jsonl(doubled, tmp_path / "test_dup.jsonl")
    predictions = tmp_path / "predictions"
    for family, noise in (("svm", 0.3), ("bilstm", 0.6), ("transformer", 0.45)):
        write_dump(predictions, family, "val", validation, noise, rng)
        write_dump(predictions, family, "test", single, noise, rng)
        with np.load(predictions / family / "test.npz") as dump:
            scores = np.repeat(dump["scores"], 2, axis=0)
        write_dump(predictions, family, "test_dup", doubled, noise, rng, scores=scores)
    args = benchmark_args(tmp_path, ["svm", "bilstm", "transformer"])
    at = args.index("--output-dir")
    args[at:at] = [f"test_dup={tmp_path / 'test_dup.jsonl'}", "--bootstrap", "500"]
    assert evaluate_ensemble.main(args) == 0
    summary = json.loads((tmp_path / "metrics/summary.json").read_text(encoding="utf-8"))
    versus = summary["significance"]["vs_best_member"]
    widths = [r["micro_f1"]["ci_high"] - r["micro_f1"]["ci_low"] for r in versus["test"].values()]
    assert max(widths) > 0
    for system, result in versus["test"].items():
        for metric in ("micro_f1", "macro_f1"):
            assert versus["test_dup"][system][metric] == pytest.approx(result[metric])


def test_bootstrap_defaults_to_ten_thousand_resamples():
    args = evaluate_ensemble.parser().parse_args(["--predictions", "p", "--members", "a", "b"])
    assert args.bootstrap == 10000


def test_significance_table_prints_holm_p_to_four_significant_digits():
    result = {"difference": 0.35, "ci_low": 0.1, "ci_high": 0.6, "p_holm": 0.00139273969}
    lines = evaluate_ensemble.significance_markdown(
        {
            "resamples": 10000,
            "best_member": "svm",
            "vs_best_member": {"test": {"soft": {"micro_f1": result}}},
            "cross_condition": {},
        }
    )
    assert "| test | soft | +0.3500 [+0.1000, +0.6000] | 0.001393 |" in lines


def test_summary_records_the_taxonomy_it_closed_labels_with(tmp_path):
    rng = np.random.default_rng(5)
    validation = write_set(tmp_path, "val", 40, rng)
    test = write_set(tmp_path, "test", 20, rng)
    for family in ("svm", "bilstm"):
        write_dump(tmp_path / "predictions", family, "val", validation, 0.3, rng)
        write_dump(tmp_path / "predictions", family, "test", test, 0.3, rng)
    assert (
        evaluate_ensemble.main(benchmark_args(tmp_path, ["svm", "bilstm"]) + ["--bootstrap", "0"])
        == 0
    )
    summary = json.loads((tmp_path / "metrics/summary.json").read_text(encoding="utf-8"))
    assert summary["taxonomy_sha256"] == compute_file_sha256("taxonomy/taxonomy.json")


def test_summary_names_role_architecture_and_model_of_every_scored_system(tmp_path):
    sets = write_benchmark(tmp_path)
    rng = np.random.default_rng(6)
    for name, rows in sets.items():
        write_dump(tmp_path / "predictions", "naive_bayes", name, rows, 0.6, rng)
    args = benchmark_args(tmp_path, ["svm", "bilstm"]) + ["--baselines", "naive_bayes"]
    assert evaluate_ensemble.main(args + ["--bootstrap", "0"]) == 0
    summary = json.loads((tmp_path / "metrics/summary.json").read_text(encoding="utf-8"))
    benchmark = summary["benchmark"]
    assert benchmark["primary_metric"] == "micro_f1"
    systems = benchmark["systems"]
    assert set(systems) == set(summary["metrics"]["test"]) | {"leave_one_out"}
    assert systems["naive_bayes"]["role"] == systems["svm"]["role"] == "baseline"
    assert systems["hard_k"]["role"] == systems["leave_one_out"]["role"] == "new"
    assert all(entry["architecture"] and entry["model"] for entry in systems.values())
    distribution = benchmark["label_distribution"]
    assert distribution["tuning"]["n_records"] == 20 and distribution["test"]["n_labels"] == 2
    page = (tmp_path / "metrics/summary.md").read_text(encoding="utf-8")
    assert page.startswith("## Benchmark") and "| naive_bayes | baseline |" in page


def test_scored_system_without_a_benchmark_entry_is_refused(tmp_path, capsys):
    sets = write_benchmark(tmp_path)
    rng = np.random.default_rng(7)
    for name, rows in sets.items():
        write_dump(tmp_path / "predictions", "mystery", name, rows, 0.3, rng)
    args = benchmark_args(tmp_path, ["svm", "bilstm", "mystery"]) + ["--bootstrap", "0"]
    assert evaluate_ensemble.main(args) == 1
    assert last_error(capsys).startswith("No benchmark entry for ['mystery']")
    assert not (tmp_path / "metrics/summary.json").exists()


def benchmark_summary():
    entry = {"role": "new", "architecture": "NALTRA vote", "model": "m"}
    distribution = {"n_records": 20, "n_labels": 2, "all_negative_accuracy": 0.25}
    return {
        "metrics": {
            "test": {
                "svm": {"micro_f1": 0.5, "macro_f1": 0.9},
                "hard_closed": {"hmicro_f1": 0.8},
                "soft": {"micro_f1": 0.7, "macro_f1": 0.1},
                "hard_k": {"micro_f1": 0.7},
            }
        },
        "leave_one_out": {"test": {"svm": 0.9}},
        "benchmark": {
            "primary_metric": "micro_f1",
            "secondary_metrics": ["macro_f1", "hmicro_f1"],
            "systems": {
                "svm": {"role": "baseline", "architecture": "linear SVM", "model": "svc"},
                "hard_closed": entry,
                "soft": entry,
                "hard_k": entry,
                "leave_one_out": entry,
            },
            "label_distribution": {"tuning": distribution, "test": distribution},
        },
    }


def test_benchmark_table_ranks_systems_by_the_primary_metric():
    lines = evaluate_ensemble.markdown(benchmark_summary()).splitlines()
    assert lines[0] == "## Benchmark"
    table = lines[: lines.index("## test")]
    rows = [line.split(" | ")[:2] for line in table if line[2:3].isdigit() or line[:4] == "| - "]
    # Ties share a competition rank; the ablation and hard_closed stay unranked.
    assert rows == [
        ["| 1", "soft"],
        ["| 1", "hard_k"],
        ["| 3", "svm"],
        ["| -", "hard_closed"],
        ["| -", "leave_one_out(-svm)"],
    ]
    assert "| 3 | svm | baseline | linear SVM | svc | 0.5000 | 0.9000 | - |" in table
    assert "| all_negative_accuracy | 0.2500 | 0.2500 |" in table


def test_benchmark_points_to_paired_difference_intervals_only_with_significance():
    summary = benchmark_summary()
    assert "interval" not in evaluate_ensemble.markdown(summary).lower()
    summary["significance"] = {
        "resamples": 10,
        "best_member": "svm",
        "vs_best_member": {},
        "cross_condition": {},
    }
    lines = evaluate_ensemble.benchmark_markdown(summary)
    assert any("Paired-difference intervals" in line for line in lines)


def test_a_test_set_named_tuning_is_refused(tmp_path):
    # Its label distribution would overwrite the tuning set's in summary.json.
    rng = np.random.default_rng(8)
    for name in ("val", "tuning"):
        rows = write_set(tmp_path, name, 20, rng)
        for family in ("svm", "bilstm"):
            write_dump(tmp_path / "predictions", family, name, rows, 0.3, rng)
    with pytest.raises(SystemExit) as result:
        evaluate_ensemble.main(benchmark_args(tmp_path, ["svm", "bilstm"], test="tuning"))
    assert result.value.code == 2


@pytest.mark.parametrize(
    "field, value",
    [
        ("primary_metric", "accuracy"),
        ("primary_metric", ["micro_f1"]),
        ("primary_metric", None),
        ("secondary_metrics", "macro_f1"),
        ("secondary_metrics", ["macro_f1", "accuracy"]),
        ("secondary_metrics", [1]),
    ],
)
def test_benchmark_metrics_must_be_ones_the_summary_produces(
    tmp_path, capsys, monkeypatch, field, value
):
    write_benchmark(tmp_path)
    config = load_yaml(evaluate_ensemble.BENCHMARK) | {field: value}
    patched = tmp_path / "benchmark.yaml"
    patched.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(evaluate_ensemble, "BENCHMARK", patched)
    args = benchmark_args(tmp_path, ["svm", "bilstm"]) + ["--bootstrap", "0"]
    assert evaluate_ensemble.main(args) == 1
    assert last_error(capsys).startswith(f"{patched}: {field} ")
    assert not (tmp_path / "metrics/summary.json").exists()


def test_tuning_breaks_ties_toward_the_lowest_threshold_and_the_largest_k():
    truth = np.array([[1, 0]])
    # 0.2 and 0.4 both select only the true label (micro-F1 1.0); 0.8 selects nothing.
    threshold, f1 = evaluate_ensemble.tune_threshold(
        truth, np.array([[0.6, 0.1]]), grid=np.array([0.2, 0.4, 0.8])
    )
    assert (threshold, f1) == (0.2, 1.0)
    # 2-of-3 and 3-of-3 both keep only the true label; 1-of-3 adds a false positive.
    k, f1 = evaluate_ensemble.tune_k(truth, np.array([[1.0, 1 / 3]]), members=3)
    assert (k, f1) == (3, 1.0)
