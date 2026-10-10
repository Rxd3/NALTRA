"""Warmed-up latency, throughput and size benchmark over trained artifacts."""

from __future__ import annotations

import importlib.metadata
import json
from types import SimpleNamespace

import pytest
from scripts import run_benchmark
from tests.test_classical_models import config, records

from naltra.data.loader import save_jsonl
from naltra.data.manifest import compute_file_sha256, get_environment_metadata
from naltra.evaluation.latency import latency_profile
from naltra.models.svm import SVMModel
from naltra.utils.provenance import commit


def test_latency_profile_discards_warmup_and_reports_percentiles() -> None:
    calls = []
    profile = latency_profile(lambda item: calls.append(item), ["a", "b", "c", "d"], warmup=2)
    assert calls == ["a", "b", "a", "b", "c", "d"]
    assert profile["samples"] == 4
    assert 0 <= profile["p50_ms"] <= profile["p95_ms"] <= profile["p99_ms"]
    with pytest.raises(ValueError):
        latency_profile(lambda item: None, [], warmup=0)


def test_latency_profile_cycles_inputs_to_reach_requested_warmup() -> None:
    calls = []
    profile = latency_profile(lambda item: calls.append(item), ["a", "b"], warmup=10)
    assert calls == ["a", "b"] * 6
    assert profile["warmup"] == 10 and profile["samples"] == 2


def test_benchmark_reports_latency_throughput_and_size(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    model = SVMModel(config())
    model.train(records("train"))
    model.save(tmp_path / "artifacts/svm")
    save_jsonl(records("test"), tmp_path / "test.jsonl")
    code = run_benchmark.main(
        [
            "--artifacts",
            str(tmp_path / "artifacts"),
            "--models",
            "svm",
            "--set",
            f"fixture={tmp_path / 'test.jsonl'}",
            "--samples",
            "5",
            "--warmup",
            "1",
            "--batch-size",
            "4",
            "--output-dir",
            str(tmp_path / "bench"),
        ]
    )
    assert code == 0
    summary = json.loads((tmp_path / "bench/benchmark_summary.json").read_text(encoding="utf-8"))
    svm = summary["models"]["svm"]
    assert svm["single"]["samples"] == 5
    assert svm["batched"]["docs_per_second"] > 0
    assert svm["artifact_bytes"] > 0 and svm["parameters"] > 0
    assert summary["hardware"]["python"]
    assert summary["eval_set"] == "fixture"
    assert summary["eval_set_path"] == "test.jsonl"
    written = (tmp_path / "bench/benchmark_summary.json").read_text(encoding="utf-8")
    assert json.dumps(str(tmp_path))[1:-1] not in written  # no local path, as JSON escapes it
    assert summary["eval_set_sha256"] == compute_file_sha256(tmp_path / "test.jsonl")
    assert (summary["samples"], summary["warmup"], summary["batch_size"]) == (5, 1, 4)
    assert summary["requested_device"] == "auto"
    assert svm["device"] == "cpu" and svm["single"]["warmup"] == 1
    assert svm["artifact_sha256"] == compute_file_sha256(tmp_path / "artifacts/svm/naltra.json")
    assert len(svm["artifact_files_sha256"]) == 64
    runtime = summary["runtime"]
    for package in ("torch", "numpy"):
        assert runtime["packages"][package] == importlib.metadata.version(package)
    assert runtime["environment"]["CUDA_VISIBLE_DEVICES"] == "-1"
    assert isinstance(runtime["cuda_available"], bool) and runtime["torch_threads"] >= 1
    page = (tmp_path / "bench/benchmark_summary.md").read_text(encoding="utf-8")
    assert "| svm |" in page and f"torch {runtime['packages']['torch']}" in page


def test_missing_artifact_is_reported_as_failure(tmp_path) -> None:
    save_jsonl(records("test"), tmp_path / "test.jsonl")
    code = run_benchmark.main(
        [
            "--artifacts",
            str(tmp_path / "none"),
            "--models",
            "svm",
            "--set",
            f"fixture={tmp_path / 'test.jsonl'}",
            "--output-dir",
            str(tmp_path / "bench"),
        ]
    )
    assert code == 1


def test_summary_records_the_code_commit_and_environment(tmp_path) -> None:
    save_jsonl(records("test"), tmp_path / "test.jsonl")
    run_benchmark.main(
        [
            "--artifacts",
            str(tmp_path / "none"),
            "--set",
            f"fixture={tmp_path / 'test.jsonl'}",
            "--output-dir",
            str(tmp_path / "bench"),
        ]
    )
    summary = json.loads((tmp_path / "bench/benchmark_summary.json").read_text(encoding="utf-8"))
    assert summary["code_commit"] == commit()
    assert summary["environment"] == get_environment_metadata()


def test_script_starts_when_run_directly() -> None:
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/run_benchmark.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=root.parent,
    )
    assert result.returncode == 0, result.stderr


def test_resolved_device_reports_where_inference_actually_runs() -> None:
    from naltra.models.hybrid_knn import HybridKNNModel
    from naltra.models.kev import KevModel
    from naltra.models.naive_bayes import NaiveBayesModel

    assert run_benchmark.resolved_device(SimpleNamespace(device="cuda:0", config={})) == "cuda:0"
    # NumPy/SciPy scoring runs on the CPU whatever device was configured.
    for preference in ("cuda", "auto"):
        assert run_benchmark.resolved_device(NaiveBayesModel({"device": preference})) == "cpu"
        assert run_benchmark.resolved_device(SVMModel({"device": preference})) == "cpu"
    knn = HybridKNNModel({"device": "cuda"}, encoder=SimpleNamespace(device="cuda:0"))
    assert run_benchmark.resolved_device(knn) == "cpu+encoder:cuda:0"
    assert run_benchmark.resolved_device(HybridKNNModel({"device": "cuda"})) == "cpu"
    assert run_benchmark.resolved_device(KevModel()) == "service"
    assert run_benchmark.resolved_device(SimpleNamespace(config={"device": "cuda"})) is None


def test_families_are_found_across_several_artifact_directories(tmp_path) -> None:
    """Members trained in different runs are timed together for one results page."""
    from naltra.models.naive_bayes import NaiveBayesModel

    for family, model in (("svm", SVMModel(config())), ("naive_bayes", NaiveBayesModel(config()))):
        model.train(records("train"))
        model.save(tmp_path / f"run_{family}/{family}")
    save_jsonl(records("test"), tmp_path / "test.jsonl")
    code = run_benchmark.main(
        [
            "--artifacts",
            str(tmp_path / "run_svm"),
            str(tmp_path / "run_naive_bayes"),
            "--models",
            "naive_bayes",
            "svm",
            "--set",
            f"fixture={tmp_path / 'test.jsonl'}",
            "--samples",
            "2",
            "--warmup",
            "1",
            "--output-dir",
            str(tmp_path / "bench"),
        ]
    )
    assert code == 0
    summary = json.loads((tmp_path / "bench/benchmark_summary.json").read_text(encoding="utf-8"))
    assert set(summary["models"]) == {"naive_bayes", "svm"} and not summary["failures"]
    nb = tmp_path / "run_naive_bayes/naive_bayes/naltra.json"
    assert summary["models"]["naive_bayes"]["artifact_sha256"] == compute_file_sha256(nb)
