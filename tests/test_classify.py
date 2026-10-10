"""The classify script runs saved artifacts through the shared inference pipeline."""

from __future__ import annotations

import json
import shutil

import pytest
from scripts import classify
from tests.test_classical_models import config, records

from naltra.data.loader import save_jsonl
from naltra.models.classical import ESTIMATOR_FILE
from naltra.models.naive_bayes import NaiveBayesModel
from naltra.models.svm import SVMModel

OPTICS = "The laser light passes through the lens and every photon is counted."
ACOUSTICS = "The sound waves carry acoustic noise across the room."


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("artifacts")
    for family in (SVMModel, NaiveBayesModel):
        model = family(config())
        model.train(records("train"), records("validation"))
        model.save(root / model.model_name)
    return root


def run(capsys, artifacts, *argv: str) -> list[dict]:
    assert classify.main(["--artifacts", str(artifacts), *argv]) == 0
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def test_single_model_reports_labels_closure_language_and_ood(artifacts, capsys):
    (row,) = run(capsys, artifacts, "--text", OPTICS)
    assert row["model"] == "svm"
    assert row["labels"][0]["label"] == "optics"
    assert row["closed_labels"] == ["natural_sciences", "optics", "physical_sciences"]
    assert row["language"] == {"primary": "en", "is_code_switched": False}
    assert row["ood"]["method"] == "max_probability"
    assert isinstance(row["ood"]["is_ood"], bool)


def test_jsonl_input_is_classified_in_order_by_the_ensemble(artifacts, tmp_path, capsys):
    save_jsonl([{"text": OPTICS}, {"text": ACOUSTICS}], tmp_path / "input.jsonl")
    rows = run(
        capsys,
        artifacts,
        "--models",
        "svm",
        "naive_bayes",
        "--method",
        "soft",
        "--input",
        str(tmp_path / "input.jsonl"),
    )
    assert [row["model"] for row in rows] == ["ensemble", "ensemble"]
    assert [row["labels"][0]["label"] for row in rows] == ["optics", "acoustics"]
    assert "physical_sciences" in rows[1]["closed_labels"]
    # The voter has no fitted OOD threshold, so the pipeline reports OOD as disabled.
    assert rows[0]["ood"] == {"is_ood": False, "score": 0.0, "method": "disabled"}


def test_missing_artifact_is_a_usage_error(artifacts, capsys):
    with pytest.raises(SystemExit) as error:
        classify.main(["--artifacts", str(artifacts), "--models", "bilstm", "--text", OPTICS])
    assert error.value.code == 2
    assert "bilstm" in capsys.readouterr().err


def test_hard_vote_reports_vote_fractions_and_soft_vote_averaged_probabilities(artifacts, capsys):
    def scores(method: str) -> list[float]:
        argv = ("--models", "svm", "naive_bayes", "--method", method, "--text", OPTICS)
        (row,) = run(capsys, artifacts, *argv)
        return [item["score"] for item in row["labels"]]

    assert set(scores("hard")) == {1.0}
    assert all(0.0 < score < 1.0 for score in scores("soft"))


def test_turkish_text_is_detected_as_turkish(artifacts, capsys):
    (row,) = run(capsys, artifacts, "--text", "Lazer ışığı mercekten geçer ve her foton sayılır.")
    assert row["language"]["primary"] == "tr"


@pytest.mark.parametrize("flag", ["--input", "--text"])
def test_missing_input_file_or_blank_text_is_a_usage_error(artifacts, tmp_path, capsys, flag):
    value = {"--input": str(tmp_path / "missing.jsonl"), "--text": "  "}[flag]
    with pytest.raises(SystemExit) as error:
        classify.main(["--artifacts", str(artifacts), flag, value])
    assert error.value.code == 2
    assert "usage:" in capsys.readouterr().err


@pytest.mark.parametrize(
    "line",
    [
        "{}",
        '{"text": null}',
        '{"text": 42}',
        '["text"]',
        '{"text": "   "}',
        "{",
        "[" * 100000,
        r'{"text": "\ud800"}',
    ],
    ids=[
        "missing",
        "null",
        "numeric",
        "non-object",
        "blank",
        "invalid-json",
        "too-deep",
        "lone-surrogate",
    ],
)
def test_an_invalid_input_record_is_a_row_indexed_usage_error_before_models_load(
    artifacts, tmp_path, capsys, monkeypatch, line
):
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps({"text": OPTICS}) + "\n" + line + "\n", encoding="utf-8")
    monkeypatch.setattr(classify, "load_model", lambda *args: pytest.fail("models loaded"))
    with pytest.raises(SystemExit) as error:
        classify.main(["--artifacts", str(artifacts), "--input", str(path)])
    assert error.value.code == 2
    err = capsys.readouterr().err
    assert "usage:" in err
    assert "--input line 2" in err


def test_an_input_file_that_is_not_utf8_is_a_usage_error(artifacts, tmp_path, capsys, monkeypatch):
    path = tmp_path / "cp1254.jsonl"
    path.write_bytes('{"text": "Çağdaş kuantum fiziği"}\n'.encode("cp1254"))
    monkeypatch.setattr(classify, "load_model", lambda *args: pytest.fail("models loaded"))
    with pytest.raises(SystemExit) as error:
        classify.main(["--artifacts", str(artifacts), "--input", str(path)])
    assert error.value.code == 2
    assert "--input line 1" in capsys.readouterr().err


def test_text_with_a_lone_surrogate_is_a_usage_error_before_models_load(
    artifacts, capsys, monkeypatch
):
    monkeypatch.setattr(classify, "load_model", lambda *args: pytest.fail("models loaded"))
    with pytest.raises(SystemExit) as error:
        classify.main(["--artifacts", str(artifacts), "--text", "laser \ud800"])
    assert error.value.code == 2
    assert "--text" in capsys.readouterr().err


@pytest.mark.parametrize("content", ["", "\n  \n\t\n"], ids=["empty", "blank-lines"])
def test_an_input_file_without_texts_is_a_usage_error_before_models_load(
    artifacts, tmp_path, capsys, monkeypatch, content
):
    path = tmp_path / "input.jsonl"
    path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(classify, "load_model", lambda *args: pytest.fail("models loaded"))
    with pytest.raises(SystemExit) as error:
        classify.main(["--artifacts", str(artifacts), "--input", str(path)])
    assert error.value.code == 2
    assert "no texts" in capsys.readouterr().err


def test_an_incomplete_artifact_is_a_usage_error_naming_the_family(artifacts, tmp_path, capsys):
    shutil.copytree(artifacts / "svm", tmp_path / "svm")
    (tmp_path / "svm" / ESTIMATOR_FILE).unlink()
    with pytest.raises(SystemExit) as error:
        classify.main(["--artifacts", str(tmp_path), "--text", OPTICS])
    assert error.value.code == 2
    err = capsys.readouterr().err
    assert "usage:" in err
    assert "svm" in err and ESTIMATOR_FILE in err
