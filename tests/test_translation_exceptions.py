"""Effect of the release's known translation exceptions on the reported scores."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts import evaluate_ensemble, translation_exceptions
from scripts.predict_all import artifact_hashes
from tests.test_make_figures import FAMILIES, benchmark
from tests.test_provenance import absolute_paths

from naltra.data.loader import load_jsonl
from naltra.data.manifest import compute_file_sha256

NONE = {"alignment_text_mismatch": [], "qa_failed": []}
FLAGGED = {"alignment_text_mismatch": ["cordis:test:0:tr"], "qa_failed": ["cordis:test:3:tr"]}


def write_artifact(artifact: Path, issues: dict) -> None:
    artifact.write_text(json.dumps({"metadata": {"translation_issues": issues}}), encoding="utf-8")


def bound(tmp_path, issues: dict) -> Path:
    """Score the fixture with svm's sidecars sealed to a real artifact, as predict_all does."""
    args = benchmark(tmp_path)
    artifact = tmp_path / "artifact/naltra.json"
    artifact.parent.mkdir()
    write_artifact(artifact, issues)
    (artifact.parent / "estimator.joblib").write_bytes(b"weights")
    for name in ("val", "test"):
        sidecar = tmp_path / "predictions" / "svm" / f"{name}.json"
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        sidecar.write_text(json.dumps(record | artifact_hashes(artifact.parent)), encoding="utf-8")
    code = evaluate_ensemble.main(
        [*args[2:4], "--members", *FAMILIES, "--tune-sets", f"val={tmp_path / 'val.jsonl'}"]
        + [*args[4:6], "--bootstrap", "0", "--output-dir", str(tmp_path / "metrics")]
    )
    assert code == 0
    return artifact


def diagnose(tmp_path, artifact: Path) -> int:
    return translation_exceptions.main(
        [
            "--summary",
            str(tmp_path / "metrics" / "summary.json"),
            "--predictions",
            str(tmp_path / "predictions"),
            "--sets",
            f"test={tmp_path / 'test.jsonl'}",
            "--artifact",
            str(artifact),
            "--output",
            str(tmp_path / "exceptions.json"),
        ]
    )


def run(tmp_path, issues: dict) -> dict:
    assert diagnose(tmp_path, bound(tmp_path, issues)) == 0
    return json.loads((tmp_path / "exceptions.json").read_text(encoding="utf-8"))


def test_exception_projects_are_split_out_by_pair_id(tmp_path) -> None:
    report = run(tmp_path, FLAGGED)
    rows = load_jsonl(tmp_path / "test.jsonl")
    flagged = {rows[0]["pair_id"], rows[3]["pair_id"]}
    assert report["exception_pairs"] == 2
    svm = report["sets"]["test"]["svm"]
    assert svm["all"]["records"] == len(rows)
    assert svm["without_exceptions"]["records"] == len(rows) - len(flagged)
    assert svm["exceptions_only"]["records"] == len(flagged)
    assert 0 <= svm["without_exceptions"]["micro_f1"] <= 1


def test_no_exceptions_leaves_scores_unchanged(tmp_path) -> None:
    report = run(tmp_path, NONE)
    svm = report["sets"]["test"]["svm"]
    assert svm["all"] == svm["without_exceptions"] and svm["exceptions_only"]["records"] == 0


def test_report_records_the_artifact_summary_and_dumps_it_used(tmp_path) -> None:
    report = run(tmp_path, FLAGGED)
    artifact, summary = tmp_path / "artifact/naltra.json", tmp_path / "metrics" / "summary.json"
    dumps = json.loads(summary.read_text(encoding="utf-8"))["dumps"]
    assert report["artifact"] == {
        "path": "naltra.json",
        "sha256": compute_file_sha256(artifact),
        "artifact_files_sha256": dumps["svm/test"]["artifact_files_sha256"],
        "family": "svm",
    }
    assert report["summary"] == {"path": "summary.json", "sha256": compute_file_sha256(summary)}
    assert absolute_paths(report, tmp_path) == []
    assert report["inputs"]["test"] == {
        "input_sha256": compute_file_sha256(tmp_path / "test.jsonl"),
        "npz_sha256": {family: dumps[f"{family}/test"]["npz_sha256"] for family in FAMILIES},
    }


def stale(artifact: Path) -> Path:
    write_artifact(artifact, FLAGGED)
    return artifact


def unrelated(artifact: Path) -> Path:
    other = artifact.parent.parent / "other" / "naltra.json"
    other.parent.mkdir()
    write_artifact(other, FLAGGED)
    return other


def reweighted(artifact: Path) -> Path:
    (artifact.parent / "estimator.joblib").write_bytes(b"other weights")
    return artifact


@pytest.mark.parametrize("tamper", [stale, unrelated, reweighted])
def test_an_artifact_no_scored_family_came_from_is_refused(tmp_path, capsys, tamper) -> None:
    artifact = tamper(bound(tmp_path, NONE))
    capsys.readouterr()
    assert diagnose(tmp_path, artifact) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["status"] == "failed"
    assert not (tmp_path / "exceptions.json").exists()


def test_exceptions_come_from_the_same_bytes_that_are_hashed(tmp_path, monkeypatch) -> None:
    """A naltra.json swapped back to the bound bytes after parsing must not pass as bound."""
    artifact = bound(tmp_path, NONE)
    original = artifact.read_bytes()
    write_artifact(artifact, FLAGGED)
    split = translation_exceptions.scored_split

    def restore_then_split(*args):
        artifact.write_bytes(original)
        return split(*args)

    monkeypatch.setattr(translation_exceptions, "scored_split", restore_then_split)
    assert diagnose(tmp_path, artifact) == 1
    assert not (tmp_path / "exceptions.json").exists()


@pytest.mark.parametrize(
    "wrong",
    [lambda a: a.parent, lambda a: a.parent / "estimator.joblib"],
    ids=["directory", "estimator"],
)
def test_artifact_must_be_a_naltra_json(tmp_path, capsys, wrong) -> None:
    artifact = wrong(bound(tmp_path, NONE))
    capsys.readouterr()
    assert diagnose(tmp_path, artifact) == 1
    error = json.loads(capsys.readouterr().out.splitlines()[-1])["error"]
    assert f"--artifact must be a model's naltra.json, not {artifact.name}" in error
    assert not (tmp_path / "exceptions.json").exists()
