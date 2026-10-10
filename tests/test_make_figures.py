"""The paper figures render only from the exact dumps a benchmark summary scored."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scripts import evaluate_ensemble, make_figures
from tests.test_evaluate_ensemble import write_dump, write_set

from naltra.data.loader import load_jsonl, save_jsonl
from naltra.data.manifest import compute_file_sha256

FAMILIES = ("svm", "naive_bayes")
FIGURES = ("language_f1", "root_confusion", "depth_f1", "reliability")


def benchmark(tmp_path) -> list[str]:
    """Score a 40-validation / 12-test fixture with evaluate_ensemble; return figure args."""
    rng = np.random.default_rng(0)
    validation, test = write_set(tmp_path, "val", 40, rng), write_set(tmp_path, "test", 12, rng)
    for family in FAMILIES:
        write_dump(tmp_path / "predictions", family, "val", validation, 0.2, rng)
        write_dump(tmp_path / "predictions", family, "test", test, 0.2, rng)
    shared = ["--predictions", str(tmp_path / "predictions")]
    test_set = ["--test-sets", f"test={tmp_path / 'test.jsonl'}"]
    code = evaluate_ensemble.main(
        [*shared, "--members", *FAMILIES, "--tune-sets", f"val={tmp_path / 'val.jsonl'}"]
        + [*test_set, "--bootstrap", "0", "--output-dir", str(tmp_path / "metrics")]
    )
    assert code == 0
    summary = ["--summary", str(tmp_path / "metrics" / "summary.json")]
    return [*summary, *shared, *test_set, "--output-dir", str(tmp_path / "plots")]


def reorder_dump(tmp_path) -> None:
    target = tmp_path / "predictions" / "svm" / "test.npz"
    with np.load(target) as dump:
        ids, labels, scores = dump["ids"], dump["labels"], dump["scores"]
    np.savez_compressed(target, ids=ids[::-1], labels=labels, scores=scores[:, ::-1])


def edit_gold(tmp_path) -> None:
    rows = load_jsonl(tmp_path / "test.jsonl")
    save_jsonl([row | {"labels_direct": ["optics"]} for row in rows], tmp_path / "test.jsonl")


def reseal_gold(tmp_path) -> None:
    edit_gold(tmp_path)
    digest = compute_file_sha256(tmp_path / "test.jsonl")
    for family in FAMILIES:
        sidecar = tmp_path / "predictions" / family / "test.json"
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        sidecar.write_text(json.dumps(record | {"input_sha256": digest}), encoding="utf-8")


def rescore_dump(tmp_path) -> None:
    rows = load_jsonl(tmp_path / "test.jsonl")
    write_dump(tmp_path / "predictions", "svm", "test", rows, 0.9, np.random.default_rng(1))


def unscored_dump(tmp_path) -> None:
    target = tmp_path / "metrics" / "summary.json"
    summary = json.loads(target.read_text(encoding="utf-8"))
    del summary["dumps"]["naive_bayes/test"]
    target.write_text(json.dumps(summary), encoding="utf-8")


def test_figures_render_from_the_dumps_the_summary_scored(tmp_path) -> None:
    assert make_figures.main(benchmark(tmp_path)) == 0
    for name in FIGURES:
        assert (tmp_path / "plots" / f"{name}.png").stat().st_size > 0


@pytest.mark.parametrize(
    "tamper", [reorder_dump, edit_gold, reseal_gold, rescore_dump, unscored_dump]
)
def test_figures_refuse_inputs_the_summary_did_not_score(tmp_path, capsys, tamper) -> None:
    args = benchmark(tmp_path)
    tamper(tmp_path)
    capsys.readouterr()
    assert make_figures.main(args) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["status"] == "failed"
    assert not list((tmp_path / "plots").glob("*.png"))


def test_figures_refuse_a_taxonomy_the_summary_did_not_use(tmp_path) -> None:
    args = benchmark(tmp_path)
    taxonomy = json.loads(open("taxonomy/taxonomy.json", encoding="utf-8").read())
    taxonomy["version"] = "edited"
    other = tmp_path / "taxonomy.json"
    other.write_text(json.dumps(taxonomy), encoding="utf-8")
    assert make_figures.main([*args, "--taxonomy", str(other)]) == 1
    assert not any((tmp_path / "plots").glob("*.png"))
