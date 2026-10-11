"""Label coverage counting, member-pair bootstrap and error bucketing on tiny fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scripts import extra_analyses
from tests.test_evaluate_ensemble import write_dump, write_set

from naltra.data.loader import save_jsonl
from naltra.data.manifest import compute_file_sha256

# nat (/1) > phys (/1/2) > optics (/1/2/3); soc (/5) > econ (/5/6); roots are unsupported.
NODES = [
    ("nat", None, "/1", False),
    ("phys", "nat", "/1/2", True),
    ("optics", "phys", "/1/2/3", True),
    ("soc", None, "/5", False),
    ("econ", "soc", "/5/6", True),
]


def write_tree(tmp_path: Path, nodes=NODES) -> Path:
    labels = [
        {"id": i, "name": i, "parent": p, "euroscivoc_code": c, "is_direct_supported": s}
        for i, p, c, s in nodes
    ]
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps({"labels": labels}), encoding="utf-8")
    return path


def project(pid: str, direct: list[str], closed: list[str]) -> dict:
    return {"project_id": pid, "labels_direct": direct, "labels": closed}


def test_parse_codes_reads_code_and_project_columns_like_preprocessing() -> None:
    text = 'code;path;title;desc;pid\n/1/2;p;"multi\nline";d;7\n/5;p;t;d;7\nshort;row\n/1;p;t;d;\n'

    assert extra_analyses.parse_codes(text) == {"7": {"/1/2", "/5"}}


def test_tree_refuses_a_code_outside_its_parents_code_path(tmp_path: Path) -> None:
    broken = [*NODES[:2], ("optics", "phys", "/9/9/3", True)]

    with pytest.raises(ValueError, match="optics"):
        extra_analyses.load_tree(write_tree(tmp_path, broken))


def test_coverage_counts_dropped_codes_missing_nodes_roots_and_rollups(tmp_path: Path) -> None:
    tree = extra_analyses.load_tree(write_tree(tmp_path))
    closed = ["nat", "phys", "optics"]
    codes = {
        "1": {"/1/2/3", "/1/2/9"},  # dropped leaf under phys: roll-up would add phys
        "2": {"/1/2/3", "/5/6/7"},  # dropped leaf under econ: soc and econ missing
        "3": {"/5", "/5/6"},  # unsupported root code, already implied by econ
    }
    records = [
        project("1", ["optics"], closed),
        project("2", ["optics"], closed),
        project("3", ["econ"], ["soc", "econ"]),
    ]

    counts = extra_analyses.coverage_counts(records, codes, tree)

    assert counts == {
        "projects": 3,
        "projects_losing_codes": 3,
        "codes_lost": 3,
        "projects_rollup_would_add_label": 2,
        "closures_missing_nodes": 1,
        "nodes_missing": 2,
        "projects_missing_root": 1,
        "roots_missing": 1,
    }
    assert extra_analyses.implied_nodes({"/5/6/7"}, tree) == {"soc", "econ"}
    assert extra_analyses.nearest_supported("/5", tree) is None


def test_closure_summary_reports_largest_change_and_whether_ranking_holds() -> None:
    systems = {
        "svm": {"hmicro_f1_stored": 0.66, "hmicro_f1_complete": 0.65},
        "knn": {"hmicro_f1_stored": 0.54, "hmicro_f1_complete": 0.56},
    }

    summary = extra_analyses.closure_summary(systems)

    assert summary["max_abs_hmicro_f1_change"] == pytest.approx(0.02)
    assert summary["ranking_unchanged"] is True
    systems["knn"]["hmicro_f1_complete"] = 0.70
    assert extra_analyses.closure_summary(systems)["ranking_unchanged"] is False


def test_member_pairs_tests_every_zero_shot_against_every_trained_member() -> None:
    rng = np.random.default_rng(0)
    truth = rng.random((40, 3)) < 0.4
    families = extra_analyses.ZERO_SHOT + extra_analyses.TRAINED
    selected = {family: rng.random((40, 3)) < 0.4 for family in families}
    selected["jev"] = selected["transformer"]
    selected["svm"] = truth
    groups = [f"pair:{row // 2}" for row in range(40)]

    tests = extra_analyses.member_pairs(truth, selected, groups, resamples=200)

    assert len(tests) == len(extra_analyses.ZERO_SHOT) * len(extra_analyses.TRAINED)
    tie = tests["jev-transformer"]["micro_f1"]
    assert tie["difference"] == tie["ci_low"] == tie["ci_high"] == 0
    assert tests["kev-svm"]["micro_f1"]["difference"] < 0
    assert all(t["micro_f1"]["p_holm"] >= t["micro_f1"]["p_value"] for t in tests.values())


def test_frequency_buckets_split_training_support_at_70_and_200() -> None:
    masks = extra_analyses.frequency_masks(np.array([0, 69, 70, 199, 200, 900]))

    assert {name: mask.nonzero()[0].tolist() for name, mask in masks.items()} == {
        "tail_lt70": [0, 1],
        "mid_70_199": [2, 3],
        "head_ge200": [4, 5],
    }


def test_f1_by_group_scores_label_columns_and_project_rows() -> None:
    truth = np.array([[1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 1, 1], [0, 0, 1]], dtype=np.uint8)
    chosen = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0], [1, 1, 1], [0, 1, 1]], dtype=bool)
    columns = {"first": np.array([True, False, False]), "rest": np.array([False, True, True])}

    by_label = extra_analyses.f1_by_group(truth, {"m": chosen}, columns, 1, "labels")
    by_count = extra_analyses.f1_by_group(
        truth, {"m": chosen}, extra_analyses.gold_count_masks(truth), 0, "projects"
    )

    assert by_label["first"] == {"labels": 1, "gold_positives": 4, "m": pytest.approx(6 / 7)}
    assert by_label["rest"]["m"] == pytest.approx(8 / 12)
    assert by_count["1"] == {"projects": 1, "gold_positives": 1, "m": pytest.approx(2 / 3)}
    assert by_count["3"]["m"] == pytest.approx(6 / 9)
    assert by_count["4+"] == {"projects": 0, "gold_positives": 0, "m": None}


def test_top_errors_rank_missed_and_false_labels_with_their_totals() -> None:
    truth = np.array([[1, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=np.uint8)
    chosen = np.array([[0, 1, 1], [0, 0, 1], [1, 0, 0]], dtype=bool)

    errors = extra_analyses.top_errors(truth, chosen, ["a", "b", "c"], top=2)

    assert errors == {
        "missed_of_gold": {"a": "2/3", "b": "1/2"},
        "false_of_predicted": {"c": "2/2", "a": "0/1"},
    }


def test_dumps_must_be_the_ones_the_committed_summary_scored(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    rows = write_set(tmp_path, "test", 6, rng)
    predictions = tmp_path / "predictions"
    write_dump(predictions, "svm", "test", rows, 0.2, rng)
    gold = tmp_path / "test.jsonl"
    summary = {
        "test_sets": {"test": {"path": str(gold), "sha256": compute_file_sha256(gold)}},
        "dumps": {"svm/test": {"npz_sha256": compute_file_sha256(predictions / "svm/test.npz")}},
        "thresholds": {"svm": 0.5},
    }

    data = extra_analyses.load_checked(predictions, tmp_path, summary, ["svm"], "test")

    assert data["truth"].shape == data["selected"]["svm"].shape == (6, 2)
    summary["dumps"]["svm/test"]["npz_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="svm/test: local dump differs"):
        extra_analyses.load_checked(predictions, tmp_path, summary, ["svm"], "test")


def write_summary(directory: Path, summary: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return directory


def test_coverage_keeps_the_committed_file_when_prediction_dumps_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    taxonomy = write_tree(tmp_path)
    tree = extra_analyses.load_tree(taxonomy)
    train = tmp_path / "train.jsonl"
    save_jsonl([project("1", ["optics"], ["nat", "phys", "optics"])], train)
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    summary = {"members": ["svm"], "baselines": [], "thresholds": {"svm": 0.5}}
    committed = tmp_path / "label_coverage.json"
    committed.write_text('{"hierarchical_micro_f1": {"en_test": {}}}\n', encoding="utf-8")
    monkeypatch.setattr(extra_analyses, "load_tree", lambda: tree)
    monkeypatch.setattr(extra_analyses, "cordis_codes", lambda: {"1": {"/1/2/3"}})
    monkeypatch.setattr(extra_analyses, "ARCHIVE", archive)
    monkeypatch.setattr(extra_analyses, "TAXONOMY", taxonomy)
    monkeypatch.setattr(extra_analyses, "SPLITS", {"train": train})
    monkeypatch.setattr(extra_analyses, "ENSEMBLE", write_summary(tmp_path / "ensemble", summary))
    monkeypatch.setattr(extra_analyses, "OUTPUTS", {"coverage": committed})
    before = committed.read_bytes()

    code = extra_analyses.main(["coverage", "--predictions", str(tmp_path / "no_dumps")])

    assert code == 1
    assert committed.read_bytes() == before
    assert "svm/en_test" in json.loads(capsys.readouterr().out)["error"]


def test_error_analysis_reads_label_support_in_each_sets_own_column_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(0)
    nodes = [
        ("phys", None, "/1", False),
        ("acoustics", "phys", "/1/2", True),
        ("optics", "phys", "/1/3", True),
    ]
    taxonomy = write_tree(tmp_path, nodes)
    tree = extra_analyses.load_tree(taxonomy)
    train = tmp_path / "train.jsonl"
    # EN-train support: acoustics 70 (mid bucket), optics 1 (tail bucket).
    save_jsonl(
        [project(str(i), ["acoustics"], []) for i in range(70)] + [project("x", ["optics"], [])],
        train,
    )
    predictions, families = tmp_path / "predictions", ("svm", "hybrid_knn")
    test_sets, dumps = {}, {}
    for name in extra_analyses.FULL_TEST:
        rows = write_set(tmp_path, name, 5, rng)  # gold: acoustics 4 projects, optics 3
        gold = tmp_path / f"{name}.jsonl"
        test_sets[name] = {"path": str(gold), "sha256": compute_file_sha256(gold)}
        for family in families:
            write_dump(predictions, family, name, rows, 0.2, rng)
            path = predictions / family / f"{name}.npz"
            if name == "tr_test":  # the TR dumps store their label columns in reverse order
                with np.load(path) as dump:
                    ids, labels, scores = dump["ids"], dump["labels"], dump["scores"]
                np.savez_compressed(path, ids=ids, labels=labels[::-1], scores=scores[:, ::-1])
            dumps[f"{family}/{name}"] = {"npz_sha256": compute_file_sha256(path)}
    summary = {
        "members": list(families),
        "baselines": [],
        "thresholds": dict.fromkeys(families, 0.5),
        "test_sets": test_sets,
        "dumps": dumps,
    }
    monkeypatch.setattr(extra_analyses, "load_tree", lambda: tree)
    monkeypatch.setattr(extra_analyses, "TAXONOMY", taxonomy)
    monkeypatch.setattr(extra_analyses, "SPLITS", {"train": train})
    monkeypatch.setattr(extra_analyses, "ENSEMBLE", write_summary(tmp_path / "ensemble", summary))

    result = extra_analyses.error_analysis(predictions)

    for name in extra_analyses.FULL_TEST:
        buckets = result["sets"][name]["label_frequency"]
        assert buckets["tail_lt70"]["gold_positives"] == 3, name
        assert buckets["mid_70_199"]["gold_positives"] == 4, name
