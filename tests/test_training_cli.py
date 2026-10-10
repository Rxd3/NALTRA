import json

import pytest
import torch
from scripts import train_all
from tests.test_neural_models import records

from naltra.models.bilstm import BiLSTMModel
from naltra.models.neural import merge_config


@pytest.fixture
def prepared_track(monkeypatch):
    def prepare(dataset, spec):
        return {
            "train": records("train"),
            "validation": records("validation"),
            "track": "en_direct",
            "provenance": {"dataset": dataset, "track": "en_direct"},
        }

    monkeypatch.setattr(train_all, "prepare_track", prepare)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("fail_bilstm", [False, True])
def test_smoke_orchestration_and_independent_failures(
    prepared_track, tmp_path, monkeypatch, fail_bilstm
):
    if fail_bilstm:

        class BrokenModel(BiLSTMModel):
            def train(self, train_data, validation_data=None):
                raise RuntimeError("deliberate training failure")

        monkeypatch.setitem(train_all.MODEL_TYPES, "bilstm", BrokenModel)
    result = train_all.main(
        ["--datasets", "cordis_h2020", "--smoke", "--device", "cpu", "--output-dir", str(tmp_path)]
    )
    assert result == int(fail_bilstm)
    summary = json.loads((tmp_path / "training_summary.json").read_text())
    assert summary[0]["status"] == ("failed" if fail_bilstm else "success")
    assert summary[1]["status"] == "success"
    artifact = tmp_path / "cordis_h2020/en_direct_smoke/transformer/naltra.json"
    metadata = json.loads(artifact.read_text())
    assert metadata["metadata"]["smoke"] is True
    assert metadata["config"]["architecture"]["pretrained_name"] == "tiny-offline-smoke"


def test_audit_failure_prevents_training_and_is_reported(tmp_path, monkeypatch):
    def fail(*args):
        raise ValueError("stale manifest")

    monkeypatch.setattr(train_all, "prepare_track", fail)
    monkeypatch.setattr(
        train_all, "tiny_transformer", lambda *args: pytest.fail("initialized before audit")
    )
    arguments = ["--datasets", "cordis_h2020", "--smoke", "--output-dir", str(tmp_path)]
    assert train_all.main(arguments) == 1
    summary = json.loads((tmp_path / "training_summary.json").read_text())
    assert len(summary) == 2
    assert all(run["status"] == "failed" and "stale manifest" in run["error"] for run in summary)
    assert not (tmp_path / "cordis_h2020").exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--models", "kev"],
        ["--epochs", "0"],
        ["--models", "bilstm", "bilstm"],
    ],
)
def test_cli_rejects_unsupported_requests(arguments):
    with pytest.raises(SystemExit) as result:
        train_all.main(arguments)
    assert result.value.code == 2


def test_cli_offers_only_cordis_datasets():
    with pytest.raises(SystemExit) as result:
        train_all.parser().parse_args(["--datasets", "unknown"])
    assert result.value.code == 2


def test_smoke_help_names_the_encoder_hybrid_knn_still_needs():
    (smoke,) = [a for a in train_all.parser()._actions if "--smoke" in a.option_strings]
    assert "hybrid_knn" in smoke.help and "Hugging Face" in smoke.help


def test_cli_has_no_legacy_heldout_topic_option():
    with pytest.raises(SystemExit) as result:
        train_all.parser().parse_args(["--datasets", "cordis_h2020", "--heldout-topic", "optics"])
    assert result.value.code == 2


def test_smoke_records_do_not_introduce_validation_only_targets():
    track = {"train": records("train"), "validation": records("validation")}
    train, validation = train_all.smoke_records(track, 1)
    assert len(train) == len(validation) == 1
    assert all(set(record["labels"]) <= {"acoustics"} for record in validation)


@pytest.mark.parametrize("prebuilt", [False, True])
def test_cordis_track_routes_to_the_selected_audit(tmp_path, monkeypatch, prebuilt):
    from naltra.data import validation

    hashes = {"en": {split: split[0] * 64 for split in ("train", "validation", "test")}}

    def release(audit):
        def run(directory, languages):
            rows = {split: [] for split in ("train", "validation", "test")}
            result = {
                "corpora": {"en": rows},
                "manifests": {},
                "label_universe": ["optics"],
                "split_sha256": hashes,
            }
            return {**result, "audit": audit} if audit == "prebuilt_records_only" else result

        return run

    monkeypatch.setattr(validation, "validate_cordis_release", release("full"))
    monkeypatch.setattr(
        validation, "validate_prebuilt_cordis_release", release("prebuilt_records_only")
    )
    spec = {"path": str(tmp_path), "languages": ["en"]}
    if prebuilt:
        spec["release"] = "prebuilt"
    track = train_all.prepare_track("cordis_h2020", spec)
    expected = "prebuilt_records_only" if prebuilt else "full"
    assert track["provenance"]["release_audit"] == expected
    assert track["provenance"]["manifest_sha256_by_language"] == {}
    assert track["provenance"]["split_sha256"] == hashes


def test_calibration_smoke_subset_keeps_only_labels_with_enough_examples():
    def row(index, labels):
        return {"id": f"r{index}", "labels": labels}

    common = [row(i, ["a"] if i % 2 else ["b"]) for i in range(12)]
    rare = [row(100, ["a", "rare"]), row(101, ["b", "rare"])]
    track = {"train": [*rare, *common], "validation": [row(200, ["a", "rare"])]}
    train, validation = train_all.smoke_records(track, 14, min_positives=3)
    counts = {}
    for record in train:
        for label in record["labels"]:
            counts[label] = counts.get(label, 0) + 1
    assert set(counts) == {"a", "b"}
    assert all(3 <= count <= len(train) - 3 for count in counts.values())
    assert validation and all(set(r["labels"]) <= {"a", "b"} for r in validation)


@pytest.mark.parametrize("changed_release", [False, True])
def test_head_refit_cli_preserves_source_and_verifies_data_release(
    tmp_path, monkeypatch, changed_release
):
    from tests.test_neural_models import small_config

    train, validation = records("train"), records("validation")
    for record in train + validation:
        record["labels_direct"] = record["labels"]
    manifests = {"en": {"taxonomy": "fixture", "output_files": "fixture"}}
    source = tmp_path / "baseline/cordis_h2020/en_direct/bilstm"
    model = BiLSTMModel(merge_config(small_config(), {"target_field": "labels_direct"}))
    model.train(train, validation)
    model.metadata.update(
        {"smoke": False, "dataset": "cordis_h2020", "dataset_manifests": manifests}
    )
    model.save(source)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    current = {
        "en": {"taxonomy": "fixture", "output_files": "changed" if changed_release else "fixture"}
    }
    monkeypatch.setattr(
        train_all,
        "prepare_track",
        lambda *args: {
            "train": train,
            "validation": validation,
            "target_field": "labels_direct",
            "track": "en_direct",
            "provenance": {
                "dataset": "cordis_h2020",
                "track": "en_direct",
                "training_languages": ["en"],
                "dataset_manifests": current,
            },
        },
    )
    output = tmp_path / "candidate"
    arguments = [
        "--datasets",
        "cordis_h2020",
        "--models",
        "bilstm",
        "--device",
        "cpu",
        "--epochs",
        "1",
        "--refit-head-from",
        str(source.parent),
        "--loss-weighting",
        "sqrt_inverse_frequency",
        "--output-dir",
        str(output),
    ]
    assert train_all.main(arguments) == int(changed_release)
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    candidate = output / "cordis_h2020/en_direct/bilstm/naltra.json"
    assert candidate.exists() == (not changed_release)
    if not changed_release:
        payload = json.loads(candidate.read_text())
        assert payload["metadata"]["training_mode"] == "frozen_encoder_head_refit"
        assert payload["metadata"]["loss"]["weighting"] == "sqrt_inverse_frequency"
        assert payload["metadata"]["initial_artifact"]["path"] == str(source)
        assert (
            train_all.main(
                [*arguments[:-2], "--output-dir", str(tmp_path / "baseline"), "--overwrite"]
            )
            == 1
        )
        assert {p.name: p.read_bytes() for p in source.iterdir()} == before
