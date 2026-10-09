"""Real scoring and orchestration checks without pretrained model downloads."""

import copy
import json
from dataclasses import replace

import pytest
import torch
from scripts import evaluate_all as evaluate

from naltra.pipeline.thresholds import apply_thresholds
from naltra.schemas.prediction import LanguageInfo, PredictionResult


def prediction(text, scores=None):
    scores = scores or {"a": 0.8, "b": 0.4}
    return PredictionResult(
        text=text,
        model="fixture",
        language=LanguageInfo("und"),
        labels=apply_thresholds(scores),
        label_scores=scores,
    )


def test_scoring_fixed_universe_hierarchy_and_all_probabilities():
    metrics = evaluate.score_predictions(
        [{"text": "one", "labels_direct": ["a"]}],
        [prediction("one")],
        ["a", "b"],
        {"root": None, "a": "root", "b": "root"},
        10,
    )
    assert metrics["classification"]["micro_f1"] == 1.0
    assert metrics["classification"]["macro_f1"] == 0.5
    assert metrics["classification"]["mean_predicted_labels"] == 1.0
    assert metrics["classification"]["empty_prediction_rate"] == 0.0
    assert metrics["hierarchical"]["micro_f1"] == 1.0
    assert metrics["calibration"]["brier_score"] == pytest.approx(0.1)
    assert metrics["calibration"]["ece"] == pytest.approx(0.3)
    assert sum(b["count"] for b in metrics["calibration"]["reliability_bins"]) == 2


@pytest.mark.parametrize("problem", ["missing_scores", "wrong_text", "empty"])
def test_scoring_rejects_incomplete_or_misaligned_predictions(problem):
    predictions = [prediction("one")]
    if problem == "missing_scores":
        predictions = [prediction("one", {"a": 0.8})]
    elif problem == "wrong_text":
        predictions = [prediction("different")]
    else:
        predictions = []
    with pytest.raises(ValueError):
        evaluate.score_predictions(
            [{"text": "one", "labels_direct": ["a"]}],
            predictions,
            ["a", "b"],
            {"a": None, "b": None},
            10,
        )


@pytest.fixture
def release(tmp_path, monkeypatch):
    base = tmp_path / "data"
    model_dir = tmp_path / "models"
    manifests = {}
    corpora = {}
    for language, target in (("en", "a"), ("tr", "b")):
        directory = base / language
        directory.mkdir(parents=True)
        manifest = {"taxonomy": {"version": "fixture"}, "output_files": {"train.jsonl": "hash"}}
        (directory / "manifest.json").write_text(json.dumps(manifest))
        manifests[language] = manifest
        corpora[language] = {
            split: [
                {
                    "id": f"{language}:{split}",
                    "text": language,
                    "labels_direct": [target],
                    "split": split,
                }
            ]
            for split in ("validation", "test")
        }
    result = {"label_universe": ["a", "b"], "manifests": manifests, "corpora": corpora}
    monkeypatch.setattr(evaluate, "validate_cordis_release", lambda *args: result)

    class SavedModel:
        model_name = "fixture"

        def __init__(self, config):
            self.config = {
                "seed": 42,
                "target_field": "labels_direct",
                "training": {"batch_size": 2},
                "multilabel": {"threshold": 0.5, "per_label": {}},
            }
            self.metadata = {
                "dataset": "cordis_h2020",
                "smoke": False,
                "training_languages": ["en", "tr"],
                "dataset_manifests": copy.deepcopy(manifests),
            }
            self.labels = ["a", "b"]
            self.parents = {"a": None, "b": None}
            self.device = torch.device("cpu")

        def load(self, path):
            self.family = path.name

        def predict_batch(self, texts):
            return [
                replace(
                    prediction(text),
                    labels=apply_thresholds(
                        prediction(text).label_scores,
                        self.config["multilabel"]["threshold"],
                        self.config["multilabel"]["per_label"],
                    ),
                )
                for text in texts
            ]

    for family in ("bilstm", "transformer"):
        (model_dir / family).mkdir(parents=True)
        (model_dir / family / "naltra.json").write_text("{}")
        monkeypatch.setitem(evaluate.MODEL_TYPES, family, SavedModel)
    return base, model_dir, result, SavedModel


def test_cli_evaluates_real_slices_and_pooled_metrics(release, tmp_path):
    base, models, _, _ = release
    output = tmp_path / "results"
    assert (
        evaluate.main(
            [
                "--data-dir",
                str(base),
                "--model-dir",
                str(models),
                "--output-dir",
                str(output),
                "--device",
                "cpu",
                "--batch-size",
                "8",
            ]
        )
        == 0
    )
    summary = json.loads((output / "evaluation_summary.json").read_text())
    assert summary["scope"] == "full" and summary["split"] == "test"
    for result in summary["models"].values():
        slices = result["slices"]
        assert slices["en"]["classification"]["micro_f1"] == 1.0
        assert slices["tr"]["classification"]["micro_f1"] == 0.0
        assert slices["combined"]["classification"]["micro_f1"] == 0.5
        assert slices["combined"]["classification"]["macro_f1"] == pytest.approx(1 / 3)
        assert slices["combined"]["records"] == 2
        assert result["batch_size"] == 8


def test_cli_marks_subset_and_preserves_existing_results(release, tmp_path):
    base, models, _, _ = release
    output = tmp_path / "results"
    args = [
        "--data-dir",
        str(base),
        "--model-dir",
        str(models),
        "--output-dir",
        str(output),
        "--max-records",
        "1",
    ]
    assert evaluate.main(args) == 0
    assert json.loads((output / "evaluation_summary.json").read_text())["scope"] == "subset"
    with pytest.raises(SystemExit):
        evaluate.main(args)


def test_failed_audit_prevents_model_loading(tmp_path, monkeypatch):
    def fail(*args):
        raise ValueError("source checksum mismatch")

    monkeypatch.setattr(evaluate, "validate_cordis_release", fail)
    monkeypatch.setitem(
        evaluate.MODEL_TYPES, "bilstm", lambda *args: pytest.fail("loaded too early")
    )
    assert evaluate.main(["--models", "bilstm", "--output-dir", str(tmp_path)]) == 1
    assert not (tmp_path / "evaluation_summary.json").exists()


def test_model_failure_is_not_reported_as_success(release, tmp_path, monkeypatch):
    base, models, _, SavedModel = release

    class Broken(SavedModel):
        def load(self, path):
            raise ValueError("incompatible artifact")

    monkeypatch.setitem(evaluate.MODEL_TYPES, "bilstm", Broken)
    output = tmp_path / "results"
    assert (
        evaluate.main(
            ["--data-dir", str(base), "--model-dir", str(models), "--output-dir", str(output)]
        )
        == 1
    )
    summary = json.loads((output / "evaluation_summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["models"]["bilstm"]["status"] == "failed"
    assert summary["models"]["transformer"]["status"] == "success"


@pytest.mark.parametrize("problem", ["smoke", "targets", "labels", "release"])
def test_artifact_contract_is_verified(release, problem):
    base, _, data, SavedModel = release
    model = SavedModel({})
    if problem == "smoke":
        model.metadata["smoke"] = True
    elif problem == "targets":
        model.config["target_field"] = "labels"
    elif problem == "labels":
        model.labels = ["a"]
    else:
        model.metadata["dataset_manifests"]["en"]["output_files"] = {"train.jsonl": "changed"}
    with pytest.raises(ValueError):
        evaluate.check_artifact(model, data, base)


def test_threshold_search_recovers_unemitted_labels_and_matches_scoring():
    records = [{"text": "one", "labels_direct": ["b"], "split": "validation"}]
    predictions = [prediction("one", {"a": 0.1, "b": 0.4})]
    selection = evaluate.select_global_threshold(records, predictions, ["a", "b"])
    assert selection["selected_threshold"] == 0.4
    assert max(candidate["micro_f1"] for candidate in selection["candidates"]) == 1.0
    assert all(
        candidate["micro_f1"] == 0
        for candidate in selection["candidates"]
        if candidate["threshold"] > 0.4
    )
    records[0]["split"] = "test"
    with pytest.raises(ValueError, match="validation"):
        evaluate.select_global_threshold(records, predictions, ["a", "b"])


@pytest.mark.parametrize("only", [False, True])
@pytest.mark.parametrize("threshold_file", [False, True])
def test_code_switch_slices_use_saved_thresholds_and_keep_clean_pool_separate(
    release, tmp_path, monkeypatch, only, threshold_file
):
    base, models, data, _ = release
    mixed = {
        "corpora": {
            strategy: {
                split: [{"text": "English Turkce", "labels_direct": ["b"], "split": split}]
                for split in ("validation", "test")
            }
            for strategy in ("sentence_mix", "chunk_mix")
        },
        "manifests": {
            strategy: {"fixture": strategy} for strategy in ("sentence_mix", "chunk_mix")
        },
    }
    monkeypatch.setattr(evaluate, "validate_code_switch_release", lambda *args: mixed)
    output = tmp_path / "mixed"
    args = [
        "--data-dir",
        str(base),
        "--model-dir",
        str(models),
        "--output-dir",
        str(output),
        "--split",
        "validation",
        "--code-switch",
        "sentence_mix",
        "chunk_mix",
    ]
    if only:
        args.append("--code-switch-only")
    if threshold_file:
        assert (
            evaluate.main(
                [
                    "--data-dir",
                    str(base),
                    "--model-dir",
                    str(models),
                    "--split",
                    "validation",
                    "--tune-threshold",
                    "--output-dir",
                    str(tmp_path / "clean_tuning"),
                ]
            )
            == 0
        )
        args.extend(["--thresholds-file", str(tmp_path / "clean_tuning/evaluation_summary.json")])
    before = {p: p.read_bytes() for p in models.rglob("*") if p.is_file()}
    assert evaluate.main(args) == 0
    report = json.loads((output / "evaluation_summary.json").read_text())
    assert report["code_switch_manifests"] == mixed["manifests"]
    assert report["dataset_manifests"] == data["manifests"]
    for result in report["models"].values():
        assert result["thresholds"]["threshold"] == (0.4 if threshold_file else 0.5)
        assert "threshold_selection" not in result
        for strategy in ("sentence_mix", "chunk_mix"):
            metrics = result["slices"][f"code_switch_{strategy}"]
            assert metrics["records"] == 1
            assert metrics["classification"]["micro_f1"] == pytest.approx(
                2 / 3 if threshold_file else 0.0
            )
        if only:
            assert len(result["slices"]) == 2
        else:
            assert result["slices"]["combined"]["records"] == 2
    assert {p: p.read_bytes() for p in before} == before


def test_code_switch_audit_blocks_loading_models(release, tmp_path, monkeypatch):
    base, models, _, _ = release

    def fail(*args):
        raise ValueError("stale mixed source")

    monkeypatch.setattr(evaluate, "validate_code_switch_release", fail)
    monkeypatch.setitem(
        evaluate.MODEL_TYPES, "bilstm", lambda *args: pytest.fail("loaded too early")
    )
    assert (
        evaluate.main(
            [
                "--data-dir",
                str(base),
                "--model-dir",
                str(models),
                "--code-switch",
                "sentence_mix",
                "--output-dir",
                str(tmp_path / "results"),
            ]
        )
        == 1
    )
    assert not (tmp_path / "results/evaluation_summary.json").exists()


@pytest.mark.parametrize(
    "args",
    [
        ["--code-switch-only"],
        ["--code-switch", "sentence_mix", "sentence_mix"],
        ["--code-switch", "sentence_mix", "--languages", "en"],
        ["--code-switch", "sentence_mix", "--split", "validation", "--tune-threshold"],
    ],
)
def test_cli_rejects_invalid_mixed_track_selections(args, tmp_path):
    with pytest.raises(SystemExit):
        evaluate.main([*args, "--output-dir", str(tmp_path)])
    assert not (tmp_path / "evaluation_summary.json").exists()


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--split", "validation", "--max-records", "1"],
        ["--split", "validation", "--languages", "en"],
    ],
)
def test_cli_rejects_tuning_on_test_or_subsets(arguments, tmp_path):
    with pytest.raises(SystemExit):
        evaluate.main(["--tune-threshold", "--output-dir", str(tmp_path), *arguments])
    assert not (tmp_path / "evaluation_summary.json").exists()


def test_cli_tunes_validation_and_reuses_frozen_thresholds_on_test(release, tmp_path):
    base, models, _, _ = release
    common = ["--data-dir", str(base), "--model-dir", str(models)]
    output = tmp_path / "validation"
    artifact_before = {p: p.read_bytes() for p in models.rglob("*") if p.is_file()}
    assert (
        evaluate.main(
            [*common, "--split", "validation", "--tune-threshold", "--output-dir", str(output)]
        )
        == 0
    )
    source = output / "evaluation_summary.json"
    summary = json.loads(source.read_text())
    for result in summary["models"].values():
        assert result["thresholds"] == {"threshold": 0.4, "per_label": {}}
        assert result["baseline_slices"]["combined"]["classification"]["micro_f1"] == 0.5
        assert result["slices"]["combined"]["classification"]["micro_f1"] == pytest.approx(2 / 3)
    assert (
        evaluate.main(
            [*common, "--thresholds-file", str(source), "--output-dir", str(tmp_path / "test")]
        )
        == 0
    )
    test = json.loads((tmp_path / "test/evaluation_summary.json").read_text())
    for result in test["models"].values():
        assert result["thresholds"]["threshold"] == 0.4
        assert "threshold_selection" not in result
        assert result["slices"]["tr"]["classification"]["micro_f1"] == pytest.approx(2 / 3)
    assert {p: p.read_bytes() for p in artifact_before} == artifact_before
    # A threshold must not silently transfer to changed weights or another release.
    (models / "bilstm/naltra.json").write_text('{"changed": true}')
    assert (
        evaluate.main(
            [*common, "--thresholds-file", str(source), "--output-dir", str(tmp_path / "changed")]
        )
        == 1
    )
    changed = json.loads((tmp_path / "changed/evaluation_summary.json").read_text())
    assert "hashes differ" in changed["models"]["bilstm"]["error"]


def test_local_laya_uses_saved_artifact_without_neural_training(release, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from naltra.models.laya import LayaModel

    base, models, data, saved_type = release

    class SavedLaya(LayaModel):
        def __init__(self, config):
            saved_type.__init__(self, config)
            self.config.pop("training")
            self.config["inference"] = {"batch_size": 1}
            self.config["checkpoint"] = {"revision": "fixture"}
            self.config["sdk_version"] = "fixture"
            self.metadata = {
                "dataset": "cordis_h2020",
                "source_languages": ["en", "tr"],
                "dataset_manifests": copy.deepcopy(data["manifests"]),
                "model_kind": "pretrained_zero_shot",
                "training_performed": False,
            }
            self.client = SimpleNamespace(initialize=lambda: None)

        load = saved_type.load
        predict_batch = saved_type.predict_batch

    monkeypatch.setitem(evaluate.MODEL_TYPES, "laya", SavedLaya)
    directory = models / "laya"
    (directory / "encoder").mkdir(parents=True)
    (directory / "naltra.json").write_text("{}")
    (directory / "encoder/config.json").write_text("{}")
    output = tmp_path / "laya_results"
    args = [
        "--models",
        "laya",
        "--data-dir",
        str(base),
        "--model-dir",
        str(models),
        "--split",
        "validation",
        "--tune-threshold",
        "--output-dir",
        str(output),
    ]
    assert evaluate.main(args) == 0
    report = json.loads((output / "evaluation_summary.json").read_text())
    result = report["models"]["laya"]
    assert result["model_kind"] == "pretrained_zero_shot"
    assert "encoder/config.json" in result["artifact_sha256"]
    assert result["thresholds"]["threshold"] == 0.4
    # The nested encoder is part of threshold provenance, not just the root weights.
    (directory / "encoder/config.json").write_text('{"changed": true}')
    assert (
        evaluate.main(
            [
                "--models",
                "laya",
                "--data-dir",
                str(base),
                "--model-dir",
                str(models),
                "--split",
                "validation",
                "--thresholds-file",
                str(output / "evaluation_summary.json"),
                "--output-dir",
                str(tmp_path / "laya_changed"),
            ]
        )
        == 1
    )


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--models", "laya"],
        ["--models", "laya", "--split", "validation", "--languages", "en"],
        ["--models", "bilstm", "--split", "validation"],
    ],
)
def test_prepare_laya_cannot_prepare_on_test_or_other_models(args):
    with pytest.raises(SystemExit):
        evaluate.main(["--prepare-laya", *args])


def test_resume_keeps_completed_batches_and_rejects_changed_artifacts(
    release, tmp_path, monkeypatch
):
    base, models, _, saved_type = release
    output = tmp_path / "resumable"
    common = [
        "--models",
        "bilstm",
        "--data-dir",
        str(base),
        "--model-dir",
        str(models),
        "--split",
        "validation",
        "--tune-threshold",
        "--resume",
        "--output-dir",
        str(output),
    ]
    original = saved_type.predict_batch
    seen = []

    def interrupted(self, texts):
        seen.extend(texts)
        if texts == ["tr"]:
            raise RuntimeError("interrupted")
        return original(self, texts)

    monkeypatch.setattr(saved_type, "predict_batch", interrupted)
    assert evaluate.main(common) == 1
    assert seen == ["en", "tr"]
    seen.clear()

    def resumed(self, texts):
        seen.extend(texts)
        return original(self, texts)

    monkeypatch.setattr(saved_type, "predict_batch", resumed)
    assert evaluate.main([*common, "--overwrite"]) == 0
    assert seen == ["tr"]  # Cached English predictions were not recomputed.
    report = json.loads((output / "evaluation_summary.json").read_text())
    result = report["models"]["bilstm"]
    assert result["slices"]["en"]["resumed_records"] == 1
    assert result["slices"]["tr"]["resumed_records"] == 0
    assert result["thresholds"]["threshold"] == 0.4
    (models / "bilstm/naltra.json").write_text('{"changed": true}')
    assert evaluate.main([*common, "--overwrite"]) == 1
    report = json.loads((output / "evaluation_summary.json").read_text())
    assert "provenance differs" in report["models"]["bilstm"]["error"]


def test_prediction_cache_rejects_corruption_and_misalignment(tmp_path):
    from types import SimpleNamespace

    from naltra.evaluation.prediction_cache import PredictionCache

    model = SimpleNamespace(
        model_name="fixture",
        labels=["a", "b"],
        config={"multilabel": {"threshold": 0.5, "per_label": {}}},
    )
    cache = PredictionCache(tmp_path / "cache.sqlite3", "fixture/en", {"split": "validation"})
    try:
        cache.append(0, [prediction("one")])
        with pytest.raises(ValueError, match="text, model or labels differ"):
            cache.load([{"text": "different"}], model)
        cache.connection.execute("UPDATE predictions SET payload = '{}' WHERE position = 0")
        cache.connection.commit()
        with pytest.raises(ValueError, match="checksum"):
            cache.load([{"text": "one"}], model)
    finally:
        cache.close()
