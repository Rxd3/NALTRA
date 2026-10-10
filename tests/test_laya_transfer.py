"""Cross-PC cache migration preserves scores and rejects scientific-input changes."""

import copy
import json
import sqlite3
import sys
from types import SimpleNamespace

import pytest
from scripts import resume_laya_other_pc as transfer


@pytest.fixture
def saved_cache(tmp_path):
    identity = {
        "artifact_sha256": {"model.safetensors": "weights"},
        "config": {"model": "laya", "inference": {"batch_size": 1}},
        "device": "cuda",
        "hardware": "NVIDIA GeForce RTX 4050 Laptop GPU",
        "environment": {
            "python_version": "3.13.5",
            "platform": "win32",
            "package_versions": {"torch": "2.6.0+cu124", "transformers": "5.18.0"},
        },
        "code": {"scripts/evaluate_all.py": "source"},
        "split": "validation",
        "records_sha256": "records",
    }
    path = tmp_path / "prediction_cache.sqlite3"
    with sqlite3.connect(path) as c:
        c.execute("CREATE TABLE runs (key TEXT PRIMARY KEY, identity TEXT NOT NULL)")
        c.execute(
            "CREATE TABLE predictions (run TEXT, position INTEGER, payload TEXT, sha256 TEXT)"
        )
        c.execute("INSERT INTO runs VALUES (?, ?)", ("laya/en", json.dumps(identity)))
        c.execute("INSERT INTO predictions VALUES ('laya/en', 0, 'saved scores', 'checksum')")
    target = copy.deepcopy(identity)
    target["hardware"] = "NVIDIA GeForce RTX 5070"
    target["environment"]["package_versions"]["torch"] = "2.7.1+cu128"
    target["environment"]["python_version"] = "3.13.9"
    return path, identity, target


def test_runtime_transfer_preserves_scores_and_full_provenance(saved_cache):
    path, source, target = saved_cache
    transfer.transfer_identity(path, "laya/en", target)
    # An identical restart must not add another transfer segment.
    transfer.transfer_identity(path, "laya/en", target)
    with sqlite3.connect(path) as c:
        assert c.execute("SELECT payload, sha256 FROM predictions").fetchall() == [
            ("saved scores", "checksum")
        ]
        assert json.loads(c.execute("SELECT identity FROM runs").fetchone()[0]) == target
        rows = c.execute(
            "SELECT completed_records, source_identity, target_identity FROM runtime_transfers"
        ).fetchall()
        assert len(rows) == 1 and rows[0][0] == 1
        assert json.loads(rows[0][1]) == source and json.loads(rows[0][2]) == target
    report = path.parent / "evaluation_summary.json"
    report.write_text(json.dumps({"status": "success", "evaluation_code": {"files": {}}}))
    transfer.annotate_report(path.parent)
    payload = json.loads(report.read_text())
    assert payload["runtime_transfer"]["single_hardware_latency_benchmark"] is False
    assert payload["runtime_transfer"]["history"][0]["source_identity"] == source
    assert "scripts/resume_laya_other_pc.py" in payload["evaluation_code"]["files"]


@pytest.mark.parametrize("field", ["artifact_sha256", "code", "split", "records_sha256"])
def test_transfer_rejects_changed_scientific_inputs(saved_cache, field):
    path, source, target = saved_cache
    target[field] = "different"
    with pytest.raises(ValueError, match="provenance differs"):
        transfer.transfer_identity(path, "laya/en", target)
    with sqlite3.connect(path) as c:
        assert json.loads(c.execute("SELECT identity FROM runs").fetchone()[0]) == source


@pytest.mark.parametrize("field", ["config", "package", "python", "platform", "torch", "device"])
def test_transfer_rejects_other_configuration_or_runtime_changes(saved_cache, field):
    path, _, target = saved_cache
    if field == "config":
        target["config"]["inference"]["batch_size"] = 8
    elif field == "package":
        target["environment"]["package_versions"]["transformers"] = "different"
    elif field == "python":
        target["environment"]["python_version"] = "3.14.0"
    elif field == "platform":
        target["environment"]["platform"] = "linux"
    elif field == "torch":
        target["environment"]["package_versions"]["torch"] = "2.9.0+cu128"
    else:
        target["device"] = "cpu"
    with pytest.raises(ValueError):
        transfer.transfer_identity(path, "laya/en", target)


def test_transfer_requires_copied_cache_and_keeps_adapter_version(saved_cache, tmp_path):
    path, _, target = saved_cache
    with pytest.raises(ValueError, match="Copy prediction_cache"):
        transfer.transfer_identity(tmp_path / "absent.sqlite3", "laya/en", target)
    transfer.transfer_identity(path, "laya/en", target)
    with sqlite3.connect(path) as c:
        c.execute("UPDATE runtime_transfers SET adapter_sha256 = 'changed'")
    with pytest.raises(ValueError, match="adapter changed"):
        transfer.transfer_identity(path, "laya/en", target)


def test_unstarted_language_is_created_by_normal_cache(saved_cache):
    path, _, target = saved_cache
    transfer.transfer_identity(path, "laya/tr", target)
    with sqlite3.connect(path) as c:
        assert c.execute("SELECT COUNT(*) FROM runs WHERE key = 'laya/tr'").fetchone()[0] == 0


def test_transfer_cli_keeps_full_validation_and_restores_cache_adapter(saved_cache, monkeypatch):
    import scripts

    path, _, target = saved_cache
    calls = []

    class NormalCache:
        def __init__(self, actual_path, key, identity):
            assert actual_path == path and key == "laya/en" and identity == target

    def evaluate_main(args):
        calls.extend(args)
        fake.PredictionCache(path, "laya/en", target)
        (path.parent / "evaluation_summary.json").write_text(
            json.dumps({"status": "success", "evaluation_code": {"files": {}}})
        )
        return 0

    fake = SimpleNamespace(
        PredictionCache=NormalCache, resolve_path=lambda _: path.parent, main=evaluate_main
    )
    monkeypatch.setattr(scripts, "evaluate_all", fake, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="2.7.1+cu128",
            cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True),
        ),
    )
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "version_info", (3, 13, 9))
    assert transfer.main(["--output-dir", str(path.parent), "--overwrite"]) == 0
    assert fake.PredictionCache is NormalCache
    assert calls[calls.index("--models") + 1] == "laya"
    assert calls[calls.index("--split") + 1] == "validation"
    assert "--tune-threshold" in calls and "--resume" in calls
    assert "--max-records" not in calls and "--code-switch" not in calls
    report = json.loads((path.parent / "evaluation_summary.json").read_text())
    assert report["runtime_transfer"]["history"][0]["records_before_transfer"] == 1
