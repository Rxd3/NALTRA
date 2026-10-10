"""Offline prediction dumps feed the ensemble and evaluation without re-running models."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scripts import predict_all
from tests.test_classical_models import config, records

from naltra.data.loader import save_jsonl
from naltra.models.svm import SVMModel


@pytest.fixture
def artifact_and_set(tmp_path):
    model = SVMModel(config())
    model.train(records("train"), records("validation"))
    model.metadata.update({"dataset": "fixture", "track": "en_tr_direct"})
    artifacts = tmp_path / "artifacts"
    model.save(artifacts / "svm")
    rows = records("test")
    rows[0]["english_source_text"] = "laser light lens photon"
    save_jsonl(rows, tmp_path / "test.jsonl")
    return model, artifacts, rows, tmp_path


def run(artifacts, tmp_path, *extra):
    return predict_all.main(
        [
            "--artifacts",
            str(artifacts),
            "--models",
            "svm",
            "--sets",
            f"fixture_test={tmp_path / 'test.jsonl'}",
            "--output-dir",
            str(tmp_path / "out"),
            "--batch-size",
            "5",
            *extra,
        ]
    )


def test_dump_matches_direct_predictions_in_input_order(artifact_and_set):
    model, artifacts, rows, tmp_path = artifact_and_set
    assert run(artifacts, tmp_path) == 0
    dump = np.load(tmp_path / "out/svm/fixture_test.npz")
    assert list(dump["ids"]) == [row["id"] for row in rows]
    assert list(dump["labels"]) == model.labels
    assert dump["scores"].dtype == np.float32
    expected = model.predict_batch([row["text"] for row in rows])
    for row_scores, result in zip(dump["scores"], expected, strict=True):
        assert row_scores == pytest.approx([result.label_scores[x] for x in model.labels], abs=1e-6)
    sidecar = json.loads((tmp_path / "out/svm/fixture_test.json").read_text(encoding="utf-8"))
    assert sidecar["model"] == "svm"
    assert sidecar["records"] == len(rows)
    assert sidecar["metadata"]["track"] == "en_tr_direct"
    assert sidecar["threshold"] == 0.5
    assert len(sidecar["input_sha256"]) == 64


def test_only_the_record_text_is_scored(artifact_and_set):
    model, artifacts, rows, tmp_path = artifact_and_set
    run(artifacts, tmp_path)
    dump = np.load(tmp_path / "out/svm/fixture_test.npz")
    direct = model.predict(rows[0]["text"]).label_scores
    assert dump["scores"][0] == pytest.approx([direct[x] for x in model.labels], abs=1e-6)


def test_existing_dump_requires_overwrite(artifact_and_set):
    _, artifacts, _, tmp_path = artifact_and_set
    assert run(artifacts, tmp_path) == 0
    assert run(artifacts, tmp_path) == 1
    assert run(artifacts, tmp_path, "--overwrite") == 0


def test_limit_writes_a_pilot_subset(artifact_and_set):
    _, artifacts, rows, tmp_path = artifact_and_set
    assert run(artifacts, tmp_path, "--limit", "3") == 0
    dump = np.load(tmp_path / "out/svm/fixture_test.npz")
    assert list(dump["ids"]) == [row["id"] for row in rows[:3]]


def test_artifact_digest_covers_every_artifact_file(artifact_and_set):
    _, artifacts, _, tmp_path = artifact_and_set
    weights = artifacts / "svm/encoder/weights.pt"
    weights.parent.mkdir()
    weights.write_bytes(b"first")

    def sidecar():
        return json.loads((tmp_path / "out/svm/fixture_test.json").read_text(encoding="utf-8"))

    assert run(artifacts, tmp_path) == 0
    before = sidecar()
    assert set(before["artifact_files"]) == {
        "naltra.json",
        "estimator.joblib",
        "encoder/weights.pt",
    }
    weights.write_bytes(b"second")
    assert run(artifacts, tmp_path, "--overwrite") == 0
    after = sidecar()
    assert after["artifact_sha256"] == before["artifact_sha256"]
    assert after["artifact_files_sha256"] != before["artifact_files_sha256"]
    assert (
        after["artifact_files"]["encoder/weights.pt"]
        != before["artifact_files"]["encoder/weights.pt"]
    )


@pytest.mark.parametrize("extra", [[], ["--limit", "3"]])
def test_empty_input_fails_without_writing(extra, artifact_and_set):
    _, artifacts, _, tmp_path = artifact_and_set
    save_jsonl([], tmp_path / "test.jsonl")
    assert run(artifacts, tmp_path, *extra) == 1
    assert not (tmp_path / "out/svm").exists()


def test_missing_artifact_fails_without_writing(tmp_path):
    save_jsonl(records("test"), tmp_path / "test.jsonl")
    assert run(tmp_path / "missing", tmp_path) == 1
    assert not (tmp_path / "out/svm/fixture_test.npz").exists()


@pytest.mark.parametrize("arguments", [["--sets", "unknown_set"], ["--models", "unknown_model"]])
def test_cli_rejects_unknown_choices(arguments, tmp_path):
    with pytest.raises(SystemExit) as result:
        predict_all.main(["--artifacts", str(tmp_path), *arguments])
    assert result.value.code == 2


def test_service_member_needs_no_artifact(tmp_path, monkeypatch):
    import httpx
    from tests.test_kev_model import answers, client

    from naltra.models.kev import KevModel

    calls = []

    def serve(request):
        calls.append(1)
        return httpx.Response(200, json=answers({"acoustics": 0.7, "optics": 0.2}))

    def factory():
        return KevModel(client(serve), {"supported_labels": ["acoustics", "optics"]})

    monkeypatch.setitem(predict_all.SERVICE_TYPES, "kev", factory)
    save_jsonl(records("test"), tmp_path / "test.jsonl")
    code = predict_all.main(
        [
            "--artifacts",
            str(tmp_path / "unused"),
            "--models",
            "kev",
            "--sets",
            f"fixture_test={tmp_path / 'test.jsonl'}",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert code == 0
    dump = np.load(tmp_path / "out/kev/fixture_test.npz")
    assert dump["scores"].shape == (len(records("test")), 2)
    assert len(calls) == len(records("test"))
    sidecar = json.loads((tmp_path / "out/kev/fixture_test.json").read_text(encoding="utf-8"))
    assert sidecar["artifact_sha256"] is None and len(sidecar["config_sha256"]) == 64
    assert sidecar["artifact_files_sha256"] is None and sidecar["artifact_files"] is None


def test_pretrained_member_loads_and_hashes_its_artifact(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from naltra.data.manifest import compute_file_sha256
    from naltra.models.laya import LayaModel
    from naltra.models.laya.client import CHECKPOINT_FILES, LayaClient

    def answer(texts, questions, **kwargs):
        choice = {"type": "choice", "probabilities": {"A": 0.7, "B": 0.3}}
        return [{"answers": {label: choice for label in questions}} for _ in texts]

    engine = SimpleNamespace(device="cpu", predict_batch=answer)
    monkeypatch.setattr(LayaClient, "initialize", lambda self: setattr(self, "agent", engine))
    checkpoint = tmp_path / "checkpoint"
    for name in CHECKPOINT_FILES:
        (checkpoint / name).parent.mkdir(parents=True, exist_ok=True)
        (checkpoint / name).write_text("fixture", encoding="utf-8")
    model = LayaModel({"device": "cpu", "supported_labels": ["acoustics", "optics"]})
    model.client = LayaClient(model.config, checkpoint_dir=checkpoint)
    model.save(tmp_path / "artifacts/laya")
    save_jsonl(records("test"), tmp_path / "test.jsonl")
    argv = ["--artifacts", str(tmp_path / "artifacts"), "--models", "laya", "--device", "cpu"]
    sets = ["--sets", f"fixture_test={tmp_path / 'test.jsonl'}"]
    assert predict_all.main([*argv, *sets, "--output-dir", str(tmp_path / "out")]) == 0
    dump = np.load(tmp_path / "out/laya/fixture_test.npz")
    assert dump["scores"].shape == (len(records("test")), 2)
    sidecar = json.loads((tmp_path / "out/laya/fixture_test.json").read_text(encoding="utf-8"))
    assert sidecar["artifact_sha256"] == compute_file_sha256(
        tmp_path / "artifacts/laya/naltra.json"
    )


A = "en_validation=a.jsonl"
B = "en_validation=b.jsonl"
COMMON = ["--summary", "s.json", "--predictions", "p"]


@pytest.mark.parametrize(
    "module, argv",
    [
        ("predict_all", ["--artifacts", "x", "--sets", A, B]),
        (
            "evaluate_ensemble",
            [
                "--predictions",
                "p",
                "--members",
                "svm",
                "naive_bayes",
                "--tune-sets",
                A,
                B,
                "--test-sets",
                "en_test",
            ],
        ),
        (
            "evaluate_ensemble",
            [
                "--predictions",
                "p",
                "--members",
                "svm",
                "naive_bayes",
                "--tune-sets",
                "tr_validation",
                "--test-sets",
                "en_test",
                "en_test=t.jsonl",
            ],
        ),
        (
            "evaluate_ensemble",
            [
                "--predictions",
                "p",
                "--members",
                "svm",
                "naive_bayes",
                "--tune-sets",
                A,
                "--test-sets",
                "en_validation=c.jsonl",
            ],
        ),
        ("make_figures", [*COMMON, "--test-sets", A, B]),
        ("build_dashboard", [*COMMON, "--test-sets", A, B]),
        ("translation_exceptions", [*COMMON, "--artifact", "x", "--sets", A, B]),
    ],
    ids=["predict", "tune", "test", "tune-and-test", "figures", "page", "exceptions"],
)
def test_repeated_set_names_are_usage_errors(module, argv, tmp_path, capsys, monkeypatch):
    import importlib

    script = importlib.import_module(f"scripts.{module}")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as result:
        script.main(argv)
    assert result.value.code == 2
    assert "en_" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("script", ["predict_all.py", "evaluate_ensemble.py"])
def test_scripts_start_when_run_directly(script):
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / script), "--help"],
        capture_output=True,
        text=True,
        cwd=root.parent,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", ["x/test", r"x\test", ""])
def test_set_names_cannot_contain_path_separators(name, tmp_path):
    with pytest.raises(SystemExit) as result:
        predict_all.main(["--artifacts", str(tmp_path), "--sets", f"{name}={tmp_path / 't.jsonl'}"])
    assert result.value.code == 2
