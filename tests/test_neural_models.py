"""Real, small neural optimization and artifact tests; no Hub downloads."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scripts.train_all import tiny_transformer
from torch import nn

from naltra.models.bilstm import BiLSTMModel
from naltra.models.neural import select_device
from naltra.models.transformer import TransformerModel


@pytest.fixture(autouse=True)
def small_cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def records(split: str) -> list[dict]:
    texts = [
        "New science research",
        "Football teams play optics",
        "Science helps optics",
        "Yeni bilim araştırması",
    ]
    labels = [
        ["acoustics"],
        ["optics"],
        ["acoustics", "optics"],
        ["acoustics"],
    ]
    return [
        {
            "id": f"{split}:{index}",
            "text": f"{text} {split}",
            "labels": target,
            "language": "tr" if index == 3 else "en",
            "source": "fixture",
            "license": "MIT",
            "split": split,
        }
        for index, (text, target) in enumerate(zip(texts, labels, strict=True))
    ]


def small_config() -> dict:
    return {
        "device": "cpu",
        "architecture": {
            "embedding_dim": 8,
            "hidden_dim": 8,
            "dropout": 0,
            "max_vocab": 100,
            "max_length": 16,
        },
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "learning_rate": 0.01,
            "gradient_accumulation_steps": 3,
        },
        "multilabel": {"threshold": 0.0},
    }


@pytest.mark.parametrize("family", ["bilstm", "transformer"])
def test_training_updates_weights_and_roundtrips_offline(family, tmp_path, monkeypatch):
    train, validation = records("train"), records("validation")
    if family == "bilstm":
        model = BiLSTMModel(small_config())
        original_initialize = model._initialize
        captured = {}

        def initialize(texts):
            original_initialize(texts)
            captured.update(
                {key: value.clone() for key, value in model.network.state_dict().items()}
            )

        monkeypatch.setattr(model, "_initialize", initialize)
    else:
        defaults = TransformerModel(small_config()).config
        model = tiny_transformer(defaults, train)
        captured = {key: value.clone() for key, value in model.network.state_dict().items()}
    model.train(train, validation)
    assert model.labels == ["acoustics", "optics"]
    assert any(
        not torch.equal(value.cpu(), captured[key])
        for key, value in model.network.state_dict().items()
    )
    assert all(
        np.isfinite(epoch["train_loss"]) and np.isfinite(epoch["validation_loss"])
        for epoch in model.history
    )
    assert 0 <= model.ood_threshold <= 1
    texts = (text for text in ["Science helps optics", "Football", "bilim araştırması"])
    predictions = model.predict_batch(texts)
    assert [result.text for result in predictions] == [
        "Science helps optics",
        "Football",
        "bilim araştırması",
    ]
    assert all(set(result.label_scores) == set(model.labels) for result in predictions)
    assert all(len(result.labels) == 2 for result in predictions)
    assert model.predict_batch([]) == []
    target = tmp_path / family
    model.save(target)
    # A network download would make this test fail.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    loaded = type(model)({"device": "cpu"})
    loaded.load(target)
    actual = loaded.predict_batch([result.text for result in predictions])
    for before, after in zip(predictions, actual, strict=True):
        assert after.label_scores == pytest.approx(before.label_scores, abs=1e-6)
    assert loaded.history == model.history
    assert loaded.ood_threshold == model.ood_threshold
    if family == "bilstm":
        assert "validation" not in loaded.vocabulary
        assert "araştırması" in loaded.vocabulary
        assert "unseen" not in loaded.vocabulary


@pytest.mark.parametrize("family", ["bilstm", "transformer"])
def test_predictions_are_independent_of_padding_and_batch_composition(family):
    train = records("train")
    model = (
        BiLSTMModel(small_config())
        if family == "bilstm"
        else tiny_transformer(TransformerModel(small_config()).config, train)
    )
    model.train(train)
    text = "Football"
    alone = model.predict(text).label_scores
    batched = model.predict_batch([text, "New science research with a much longer sentence"])[
        0
    ].label_scores
    assert alone == pytest.approx(batched, abs=1e-6)
    assert model.ood_threshold is None


@pytest.mark.parametrize("family", ["bilstm", "transformer"])
def test_cached_head_refit_preserves_encoder_and_roundtrips(family, tmp_path):
    train, validation = records("train"), records("validation")
    model = (
        BiLSTMModel(small_config())
        if family == "bilstm"
        else tiny_transformer(TransformerModel(small_config()).config, train)
    )
    model.train(train, validation)
    inputs = model._encode([record["text"] for record in validation])
    with torch.inference_mode():
        features = model._features(inputs)
        assert torch.allclose(model._feature_logits(features), model._logits(inputs), atol=1e-6)
    before = {key: value.clone() for key, value in model.network.state_dict().items()}
    head_parameters = {id(parameter) for parameter in model._classifier().parameters()}
    head_names = {
        name
        for name, parameter in model.network.named_parameters()
        if id(parameter) in head_parameters
    }
    model.config["training"].update(
        {
            "epochs": 3,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "loss_weighting": "sqrt_inverse_frequency",
            "feature_batch_size": 2,
        }
    )
    model.refit_head(train, validation)
    assert all(
        torch.equal(value, before[name])
        for name, value in model.network.state_dict().items()
        if name not in head_names
    )
    assert any(
        not torch.equal(value, before[name])
        for name, value in model.network.state_dict().items()
        if name in head_names
    )
    with torch.inference_mode():
        assert torch.equal(features, model._features(inputs))
        assert torch.allclose(model._feature_logits(features), model._logits(inputs), atol=1e-6)
    assert model.ood_threshold is None
    assert model.metadata["training_mode"] == "frozen_encoder_head_refit"
    assert model.metadata["loss"]["training_records"] == len(train)
    prediction = model.predict("Science helps optics")
    model.save(tmp_path)
    loaded = type(model)({"device": "cpu"})
    loaded.load(tmp_path)
    assert loaded.predict(prediction.text).label_scores == pytest.approx(
        prediction.label_scores, abs=1e-6
    )
    assert loaded.metadata["loss"] == model.metadata["loss"]


def test_positive_weights_use_only_train_targets_and_weight_validation_consistently():
    train, validation = records("train"), records("validation")
    for record in train:
        record["labels"] = ["acoustics"]
    train[0]["labels"] = ["optics"]
    for record in validation:
        record["labels"] = ["optics"]
    model = BiLSTMModel(
        {
            **small_config(),
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "loss_weighting": "sqrt_inverse_frequency",
                "max_positive_weight": 1.5,
            },
        }
    )
    model.train(train, validation)
    assert model.metadata["loss"]["positive_counts"] == {"acoustics": 3, "optics": 1}
    assert model.metadata["loss"]["positive_weights"] == {"acoustics": 1.0, "optics": 1.5}
    loader = model._loader(validation)
    with torch.inference_mode():
        total = 0.0
        for batch in loader:
            targets = batch.pop("targets")
            total += nn.functional.binary_cross_entropy_with_logits(
                model._logits(batch), targets, pos_weight=torch.tensor([1.0, 1.5])
            ).item() * len(targets)
    assert model._validation_loss(loader) == pytest.approx(total / len(validation))


def test_early_stopping_restores_best_checkpoint(monkeypatch):
    model = BiLSTMModel(
        {**small_config(), "training": {"epochs": 8, "batch_size": 2, "patience": 2}}
    )
    captured = []

    def validation_loss(loader):
        captured.append({key: value.clone() for key, value in model.network.state_dict().items()})
        return len(captured) / 10

    monkeypatch.setattr(model, "_validation_loss", validation_loss)
    model.train(records("train"), records("validation"))
    assert len(model.history) == 3
    assert all(
        torch.equal(value, captured[0][key]) for key, value in model.network.state_dict().items()
    )


@pytest.mark.parametrize(
    "config",
    [
        {"training": {"epochs": 0}},
        {"training": {"batch_size": 0}},
        {"training": {"gradient_accumulation_steps": 0}},
        {"training": {"learning_rate": float("nan")}},
        {"device": "invalid"},
        {"architecture": {"max_vocab": 2}},
        {"architecture": {"max_length": 0}},
        {"multilabel": {"per_label": {"optics": 2}}},
    ],
)
def test_rejects_invalid_configuration(config):
    with pytest.raises(ValueError):
        BiLSTMModel(config)


def test_requires_fitted_state_and_valid_records():
    model = BiLSTMModel(small_config())
    with pytest.raises(RuntimeError, match="trained or loaded"):
        model.predict("some text")
    with pytest.raises(RuntimeError, match="unfitted"):
        model.save(Path("unused"))
    with pytest.raises(ValueError, match="empty"):
        model.train([])
    with pytest.raises(ValueError, match="other splits"):
        model.train(records("test"))
    validation = records("validation")
    validation[0]["labels"] = ["biochemistry"]
    with pytest.raises(ValueError, match="absent from training"):
        model.train(records("train"), validation)
    model.train(records("train"))
    with pytest.raises(ValueError, match="empty"):
        model.predict_batch(["valid", " \n "])
    with pytest.raises(TypeError, match="string"):
        model.predict(None)


def test_artifact_taxonomy_mismatch(tmp_path):
    model = BiLSTMModel(small_config())
    model.train(records("train"))
    model.save(tmp_path)
    path = tmp_path / "naltra.json"
    metadata = json.loads(path.read_text())
    metadata["taxonomy"]["taxonomy_version"] = "bad"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="taxonomy"):
        BiLSTMModel(small_config()).load(tmp_path)


def test_cuda_unavailable_is_actionable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device("auto").type == "cpu"
    with pytest.raises(RuntimeError, match="--device cpu"):
        select_device("cuda")


@pytest.mark.parametrize("key", ["id", "pair_id", "project_id", "text"])
def test_train_and_head_refit_reject_record_project_and_content_leakage(key):
    train, validation = records("train"), records("validation")
    config = small_config()
    config["training"]["gradient_accumulation_steps"] = 1
    model = BiLSTMModel(config)
    model.train(train, validation)
    validation[0][key] = train[0][key] = train[0].get(key, "cordis:1")
    for fit in (model.train, model.refit_head):
        with pytest.raises(ValueError, match=f"{key} contamination"):
            fit(train, validation)


def test_custom_taxonomy_is_hashed_and_cannot_change_after_training(tmp_path):
    from naltra.data.manifest import REPO_ROOT

    custom = tmp_path / "custom_taxonomy.json"
    custom.write_text(
        (REPO_ROOT / "taxonomy/taxonomy.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "label_map.json").write_text(
        (REPO_ROOT / "taxonomy/label_map.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    model = BiLSTMModel(small_config(), taxonomy_path=custom)
    model.train(records("train"))
    model.save(tmp_path / "artifact")
    custom.write_text(custom.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="Taxonomy changed"):
        model.save(tmp_path / "other_artifact")
    with pytest.raises(ValueError, match="taxonomy"):
        BiLSTMModel(small_config(), taxonomy_path=custom).load(tmp_path / "artifact")


@pytest.mark.parametrize("family", ["bilstm", "transformer"])
def test_cordis_targets_exclude_structural_ancestors(family, tmp_path):
    from naltra.data.preprocessing import hierarchy_closure

    config = {**small_config(), "target_field": "labels_direct"}
    model = BiLSTMModel(config)
    train, validation = records("train"), records("validation")
    for row in train + validation:
        row["labels_direct"] = list(row["labels"])
        row["labels"] = sorted(hierarchy_closure(row["labels_direct"], model.parents))
    if family == "transformer":
        model = tiny_transformer(TransformerModel(config).config, train)
    model.train(train, validation)
    assert model.labels == ["acoustics", "optics"]
    assert "natural_sciences" not in model.predict("Scientific research").label_scores
    model.save(tmp_path)
    reloaded = type(model)({"device": "cpu"})
    reloaded.load(tmp_path)
    assert reloaded.config["target_field"] == "labels_direct"
    assert set(reloaded.predict("Scientific research").label_scores) == {"acoustics", "optics"}
