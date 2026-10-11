"""Label coverage, member-pair and drop bootstraps, errors and TR screens on tiny fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scripts import extra_analyses
from tests.test_evaluate_ensemble import write_dump, write_set

from naltra.data.loader import save_jsonl
from naltra.data.manifest import compute_file_sha256
from naltra.evaluation.bootstrap import paired_bootstrap

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

    # Unpruned: /1/2/9 (project 1) and /5, /5/6, /5/6/7 (project 2) are missing; the two codes
    # outside the active tree count there only.
    assert counts == {
        "projects": 3,
        "projects_losing_codes": 3,
        "codes_lost": 3,
        "projects_rollup_would_add_label": 2,
        "closures_missing_nodes": 1,
        "nodes_missing": 2,
        "projects_missing_root": 1,
        "roots_missing": 1,
        "closures_missing_unpruned_paths": 2,
        "unpruned_paths_missing": 4,
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


def test_every_output_records_the_commit_taken_before_the_first_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = {name: tmp_path / f"{name}.json" for name in extra_analyses.OUTPUTS}
    calls = []

    def described() -> str:
        calls.append(None)
        # Like git describe: once a result file is written, the tree is dirty.
        return "abc1234-dirty" if any(path.exists() for path in outputs.values()) else "abc1234"

    monkeypatch.setattr(extra_analyses, "commit", described)
    monkeypatch.setattr(extra_analyses, "OUTPUTS", outputs)
    builders = ("label_coverage", "pairwise", "error_analysis", "drops", "translation_checks")
    for builder in (*builders, "mt_quality"):
        monkeypatch.setattr(extra_analyses, builder, lambda *args: {"value": 1})

    assert extra_analyses.main([]) == 0

    assert len(calls) == 1
    defaults = [path for name, path in outputs.items() if name not in extra_analyses.OPT_IN]
    recorded = [json.loads(path.read_text(encoding="utf-8")) for path in defaults]
    assert [payload["code_commit"] for payload in recorded] == ["abc1234"] * len(builders)
    assert not any(outputs[name].exists() for name in extra_analyses.OPT_IN)


def drop_sets(rng: np.random.Generator) -> tuple[dict, dict]:
    """EN and condition selections of every drop member on 40 projects, paired by pair_id.

    SVM gets the most condition noise; bilstm repeats SVM exactly; the condition lists the
    projects in reverse order.
    """
    pairs = [f"cordis:{index}" for index in range(40)]
    truth = rng.random((40, 3)) < 0.4
    en = {f: truth ^ (rng.random(truth.shape) < 0.1) for f in extra_analyses.DROP_MEMBERS}
    shifted = {f: truth ^ (rng.random(truth.shape) < 0.3) for f in extra_analyses.DROP_MEMBERS}
    shifted["svm"] = truth ^ (rng.random(truth.shape) < 0.5)
    en["bilstm"], shifted["bilstm"] = en["svm"], shifted["svm"]
    order = np.arange(40)[::-1]
    reference = {"rows": [{"pair_id": p} for p in pairs], "truth": truth, "selected": en}
    condition = {
        "rows": [{"pair_id": pairs[i]} for i in order],
        "truth": truth[order],
        "selected": {family: chosen[order] for family, chosen in shifted.items()},
    }
    return reference, condition


def test_drop_tests_resample_projects_jointly_across_sets_and_members() -> None:
    reference, condition = drop_sets(np.random.default_rng(0))
    truth, pairs = reference["truth"], [row["pair_id"] for row in reference["rows"]]
    shifted = {f: chosen[::-1] for f, chosen in condition["selected"].items()}  # EN row order

    result = extra_analyses.drop_tests(reference, condition, resamples=500)

    for family in extra_analyses.DROP_MEMBERS:
        alone = paired_bootstrap(
            truth, reference["selected"][family], truth, shifted[family], 500, groups=pairs
        )["micro_f1"]
        drop = result["drops"][family]
        for key in ("difference", "ci_low", "ci_high", "p_value"):
            assert drop[key] == pytest.approx(alone[key]), (family, key)
    tests = result["drop_differences"]
    assert list(tests) == [f"svm-{f}" for f in extra_analyses.DROP_MEMBERS if f != "svm"]
    same = tests["svm-bilstm"]
    assert same["difference"] == same["ci_low"] == same["ci_high"] == 0
    assert same["p_value"] == same["p_holm"] == 1
    knn = result["drops"]["svm"]["difference"] - result["drops"]["hybrid_knn"]["difference"]
    assert tests["svm-hybrid_knn"]["difference"] == pytest.approx(knn)
    assert all(t["p_holm"] >= t["p_value"] for t in tests.values())


def test_drop_tests_refuse_conditions_that_hold_other_projects() -> None:
    reference, condition = drop_sets(np.random.default_rng(0))
    condition["rows"][0] = {"pair_id": "cordis:elsewhere"}

    with pytest.raises(ValueError, match="same projects"):
        extra_analyses.drop_tests(reference, condition, resamples=10)


def test_drops_must_reproduce_the_committed_cross_condition_differences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference, condition = drop_sets(np.random.default_rng(0))
    sets = {"en_test": reference} | dict.fromkeys(extra_analyses.DROP_CONDITIONS, condition)

    def drop(family: str) -> float:
        en, shifted = reference["selected"][family], condition["selected"][family]
        return extra_analyses.micro_f1(reference["truth"], en) - extra_analyses.micro_f1(
            condition["truth"], shifted
        )

    cross = {
        f: {f"en_test->{c}": {"micro_f1": {"difference": drop(f)}} for c in sets if c != "en_test"}
        for f in extra_analyses.DROP_MEMBERS
    }
    summary = {
        "thresholds": dict.fromkeys(extra_analyses.DROP_MEMBERS, 0.5),
        "significance": {"cross_condition": cross},
    }
    monkeypatch.setattr(extra_analyses, "ENSEMBLE", write_summary(tmp_path / "ensemble", summary))
    monkeypatch.setattr(extra_analyses, "load_checked", lambda *args: sets[args[-1]])

    result = extra_analyses.drops(tmp_path, resamples=100)

    assert list(result["conditions"]) == list(extra_analyses.DROP_CONDITIONS)
    cross["svm"]["en_test->tr_test"]["micro_f1"]["difference"] += 0.001
    write_summary(tmp_path / "ensemble", summary)
    with pytest.raises(ValueError, match="svm en_test->tr_test"):
        extra_analyses.drops(tmp_path, resamples=100)


def test_truncation_needs_an_en_sentence_over_80_characters_kept_under_40_percent() -> None:
    def pair(en: int, tr: int) -> dict[str, str]:
        return {"en": "e" * en, "tr": "t" * tr}

    assert extra_analyses.truncated([pair(20, 1), pair(81, 32)])  # 32 < 0.4 * 81
    assert not extra_analyses.truncated([pair(81, 33)])
    assert not extra_analyses.truncated([pair(80, 1)])


def test_repetition_needs_a_doubled_4gram_that_the_en_source_does_not_double() -> None:
    en = "Design and production of the distributed system of scientific collections."
    copied = "distributed system of scientific collections"

    assert extra_analyses.mt_repetition("Tasarim ve uretim sureci tasarim ve uretim sureci.", en)
    assert extra_analyses.mt_repetition(f"{copied}, {copied.upper()} kurulur.", en)
    assert not extra_analyses.mt_repetition("veri veri analiz analiz bir iki uc bir iki uc", en)
    assert not extra_analyses.mt_repetition(f"{copied} {copied}", f"See {copied} {copied}.")


def write_processed(directory: Path, language: str, split: str, records: list[dict]) -> None:
    path = directory / language / f"{split}.jsonl"
    save_jsonl(records, path)
    manifest = path.parent / "manifest.json"
    files = (
        json.loads(manifest.read_text(encoding="utf-8"))["output_files"]
        if manifest.exists()
        else {}
    )
    files[path.name] = {"sha256": compute_file_sha256(path)}
    manifest.write_text(json.dumps({"output_files": files}), encoding="utf-8")


def test_translation_checks_count_flags_per_split_on_manifest_locked_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentence = "x" * 90
    for split, count in (("train", 6), ("validation", 1), ("test", 1)):
        kept = "y" if split == "train" else sentence
        en = [{"pair_id": f"{split}:{i}", "text": sentence} for i in range(count)]
        tr = [
            {
                "id": f"{split}:{i}:tr",
                "pair_id": f"{split}:{i}",
                "text": "a b c d a b c d" if i == 1 else "tamam",
                "sentence_alignment": [{"en": sentence, "tr": kept}],
            }
            for i in range(count)
        ]
        write_processed(tmp_path, "en", split, en)
        write_processed(tmp_path, "tr", split, tr)
    monkeypatch.setattr(extra_analyses, "PROCESSED", tmp_path)

    result = extra_analyses.translation_checks()

    train = result["splits"]["train"]
    assert train["records"] == 6
    assert train["truncation"] == {
        "records": 6,
        "rate": 1.0,
        "examples": [f"train:{i}:tr" for i in range(5)],
    }
    assert train["repetition"] == {
        "records": 1,
        "rate": pytest.approx(1 / 6),
        "examples": ["train:1:tr"],
    }
    assert result["splits"]["test"]["truncation"]["records"] == 0
    assert result["splits"]["all"] == {
        "records": 8,
        "truncation": {"records": 6, "rate": 0.75},
        "repetition": {"records": 1, "rate": 0.125},
    }
    save_jsonl([{"pair_id": "test:0", "text": "changed"}], tmp_path / "en/test.jsonl")
    with pytest.raises(ValueError, match="en/test.jsonl"):
        extra_analyses.translation_checks()


def test_chrf_matches_hand_computed_scores() -> None:
    chrf = extra_analyses.chrf

    assert chrf("a b  c", "abc") == pytest.approx(100)  # whitespace is removed
    assert chrf("ab", "cd") == chrf("", "abc") == 0
    # Hypothesis "ab", reference "abc": P = (2/2 + 1/1) / 2 = 1, R = (2/3 + 1/2) / 2 = 7/12,
    # F2 = 5PR / (4P + R) = 7/11; order 3 is skipped, as the hypothesis has no 3-gram.
    assert chrf("ab", "abc") == pytest.approx(100 * 7 / 11)
    # Swapped, P = 7/12 and R = 1 give F2 = 35/40: beta = 2 weighs recall more.
    assert chrf("abc", "ab") == pytest.approx(87.5)
    # Matches are clipped: "aaaa" against "aa" gives P = (2/4 + 1/3) / 2 = 5/12, R = 1.
    assert chrf("aaaa", "aa") == pytest.approx(100 * 25 / 32)
    # One differing last character of 7: P = R = (7 - n) / (8 - n) for n = 1..6, no 7-grams.
    expected = sum((7 - n) / (8 - n) for n in range(1, 7)) / 6
    assert chrf("abcdefg", "abcdefh") == pytest.approx(100 * expected)


def test_round_trip_sample_is_seeded_sorted_and_independent_of_file_order() -> None:
    ids = [f"cordis:{index}:tr" for index in range(1000)]

    sample = extra_analyses.round_trip_sample(ids, 300)

    assert len(sample) == len(set(sample)) == 300 and set(sample) <= set(ids)
    assert sample == sorted(sample)
    assert sample == extra_analyses.round_trip_sample(reversed(ids), 300)
    assert sample != extra_analyses.round_trip_sample(ids, 300, seed=1)
    with pytest.raises(ValueError):
        extra_analyses.round_trip_sample(ids[:10], 300)


def test_language_counts_report_records_not_detected_as_the_expected_language() -> None:
    texts = ["tr", "en", "tr", "und", "en-tr", "en", "en", "en"]
    records = [{"id": f"r{index}", "text": text} for index, text in enumerate(texts)]

    result = extra_analyses.language_counts(records, "tr", lambda text: text)

    assert result == {
        "records": 8,
        "detected": {"en": 4, "en-tr": 1, "tr": 2, "und": 1},
        "not_tr": {"records": 6, "rate": 0.75, "examples": ["r1", "r3", "r4", "r5", "r6"]},
    }


def test_language_id_sums_the_splits_of_manifest_locked_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for split, texts in (("train", ["tr", "en", "tr"]), ("validation", ["tr"]), ("test", ["und"])):
        records = [{"id": f"{split}:{i}", "text": text} for i, text in enumerate(texts)]
        write_processed(tmp_path, "tr", split, records)
    monkeypatch.setattr(extra_analyses, "PROCESSED", tmp_path)

    result = extra_analyses.language_id("tr", lambda text: text)

    assert result["train"]["not_tr"]["examples"] == ["train:1"]
    assert result["all"] == {
        "records": 5,
        "detected": {"en": 1, "tr": 3, "und": 1},
        "not_tr": {"records": 2, "rate": 0.4},
    }


def test_round_trip_summary_compares_screened_records_with_the_unflagged_sample() -> None:
    scores = {"a": 20.0, "b": 35.0, "c": 50.0, "d": 60.0, "e": 70.0, "f": 10.0}
    flags = {"truncation": ["a", "f"], "repetition": []}

    result = extra_analyses.round_trip_summary(scores, ["a", "b", "c", "d", "e"], flags)

    assert result["chrf"] == {"mean": 47, "q1": 35, "median": 50, "q3": 60, "min": 20, "max": 70}
    assert result["below_40"] == {"documents": 2, "rate": 0.4}
    assert result["below_30"] == {"documents": 1, "rate": 0.2}
    screens = result["screens"]
    assert screens["truncation"] == {"documents": 2, "in_sample": 1, "mean_chrf": 15}
    assert screens["repetition"] == {"documents": 0, "in_sample": 0, "mean_chrf": None}
    assert screens["any_screen"] == screens["truncation"]
    assert screens["unflagged_sample"] == {"documents": 4, "in_sample": 4, "mean_chrf": 53.75}
    assert result["lowest"] == {"a": 20, "b": 35, "c": 50, "d": 60, "e": 70}
