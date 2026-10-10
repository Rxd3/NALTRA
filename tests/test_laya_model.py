"""Local multilingual Laya: independent topic questions, batching and portable artifacts."""

from __future__ import annotations

import json

import pytest

from naltra.models.laya import LayaModel
from naltra.models.laya.client import LayaClient


@pytest.fixture
def local_laya(tmp_path):
    from types import SimpleNamespace

    from naltra.models.laya.client import CHECKPOINT_FILES

    model = LayaModel(
        {
            "device": "cpu",
            "supported_labels": ["optics", "acoustics"],
            "inference": {"question_batch_size": 1},
        }
    )
    checkpoint = tmp_path / "checkpoint"
    for name in CHECKPOINT_FILES:
        file = checkpoint / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("fixture")
    calls = []

    def predict_batch(texts, questions, **kwargs):
        calls.append((texts, questions, kwargs))
        return [
            {
                "answers": {
                    label: {
                        "type": "choice",
                        "probabilities": {
                            "A": 0.8 if question["criteria"]["A"] in text else 0.2,
                            "B": 0.2 if question["criteria"]["A"] in text else 0.8,
                        },
                    }
                    for label, question in questions.items()
                }
            }
            for text in texts
        ]

    client = LayaClient(model.config, checkpoint_dir=checkpoint)
    client.agent = SimpleNamespace(device=model.device, predict_batch=predict_batch)
    model.client = client
    return model, calls


def test_local_laya_independent_topics_batch_order_and_pipeline(local_laya):
    from naltra.pipeline.prediction import PredictionPipeline
    from naltra.schemas.prediction import LanguageInfo

    model, calls = local_laya
    results = model.predict_batch([" optics   acoustics ", "nothing", "optics"])
    assert results[0].label_scores == {"acoustics": 0.8, "optics": 0.8}
    assert len(results[0].labels) == 2  # Multi-label probabilities need not sum to one.
    assert results[1].labels == []
    assert [item.label for item in results[2].labels] == ["optics"]
    assert all(len(group) == 1 for _, group, _ in calls)
    assert all(kwargs["batch_size"] == 1 for _, _, kwargs in calls)
    pipeline = PredictionPipeline(model=model)
    result = pipeline.predict("optics", language=LanguageInfo("tr", True))
    assert result.language.is_code_switched and result.hierarchy_paths
    assert model.predict_batch([]) == []
    with pytest.raises(ValueError, match="empty"):
        model.predict_batch(["optics", " "])


@pytest.mark.parametrize("problem", ["missing", "nan", "range", "sum", "boolean", "batch"])
def test_local_laya_rejects_broken_probabilities(local_laya, problem):
    model, _ = local_laya

    def broken(texts, questions, **kwargs):
        if problem == "batch":
            return []
        probabilities = {"A": 0.8, "B": 0.2}
        if problem == "nan":
            probabilities["A"] = float("nan")
        elif problem == "range":
            probabilities["A"] = 1.2
        elif problem == "sum":
            probabilities["B"] = 0.8
        elif problem == "boolean":
            probabilities["A"] = True
        return [
            {
                "answers": (
                    {}
                    if problem == "missing"
                    else {
                        label: {"type": "choice", "probabilities": probabilities}
                        for label in questions
                    }
                )
            }
            for _ in texts
        ]

    model.client.agent.predict_batch = broken
    with pytest.raises(ValueError):
        model.predict("optics")


def test_local_laya_portable_artifact_and_integrity(local_laya, tmp_path, monkeypatch):
    model, _ = local_laya
    artifact = tmp_path / "artifact"
    model.save(artifact)
    loaded = LayaModel({"device": "cpu"})
    loaded.load(artifact)
    assert loaded.labels == model.labels and loaded.questions == model.questions
    assert loaded.client.checkpoint_dir == artifact
    # Never download a checkpoint in unit tests; an injected engine stands in for the SDK.
    loaded.client.agent = model.client.agent
    assert loaded.predict("optics").label_scores == model.predict("optics").label_scores
    with pytest.raises(ValueError, match="overwrite"):
        model.save(artifact)
    payload = json.loads((artifact / "naltra.json").read_text())
    payload["questions"]["optics"]["criteria"]["A"] = "tampered"
    (artifact / "naltra.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="questions"):
        loaded.load(artifact)
    payload["questions"] = model.questions
    (artifact / "naltra.json").write_text(json.dumps(payload))
    (artifact / "model.safetensors").write_text("tampered")
    with pytest.raises(ValueError, match="corrupted"):
        loaded.load(artifact)


@pytest.mark.parametrize(
    "config",
    [
        {"api_key": "forbidden"},
        {"checkpoint": {"revision": "main"}},
        {"inference": {"question_batch_size": 0}},
        {"supported_labels": ["unknown"]},
        {"supported_labels": ["optics", "optics"]},
        {"multilabel": {"per_label": {"unknown": 0.3}}},
    ],
)
def test_local_laya_configuration_validation(config):
    with pytest.raises(ValueError):
        LayaModel(config)
