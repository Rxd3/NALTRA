"""Small, offline tests for the TF-IDF Naive Bayes and SVM members."""

from __future__ import annotations

import json

import pytest

from naltra.models.classical import translation_group
from naltra.models.hybrid_knn import HybridKNNModel
from naltra.models.naive_bayes import NaiveBayesModel
from naltra.models.svm import SVMModel

FAMILIES = [NaiveBayesModel, SVMModel]

# Each label owns a distinct vocabulary so a working classifier must separate them.
TOPICS = {
    "acoustics": ["sound waves acoustic noise", "ses dalgaları akustik gürültü"],
    "optics": ["laser light lens photon", "lazer ışık mercek foton"],
}


def records(split: str, per_topic: int = 4) -> list[dict]:
    rows = []
    for label, (english, turkish) in TOPICS.items():
        for index in range(per_topic):
            for language, text in (("en", english), ("tr", turkish)):
                rows.append(
                    {
                        "id": f"{split}:{label}:{index}:{language}",
                        "pair_id": f"{split}:{label}:{index}",
                        "text": f"{text} sample {index} {split}",
                        "labels": [label],
                        "labels_direct": [label],
                        "language": language,
                        "source": "fixture",
                        "license": "MIT",
                        "split": split,
                    }
                )
    both = "sound waves acoustic laser light lens"
    rows.extend(
        {
            "id": f"{split}:both:{index}",
            "text": f"{both} mixed {index} {split}",
            "labels": ["acoustics", "optics"],
            "labels_direct": ["acoustics", "optics"],
            "language": "en",
            "source": "fixture",
            "license": "MIT",
            "split": split,
        }
        for index in range(3)
    )
    return rows


def config() -> dict:
    return {"target_field": "labels_direct", "n_jobs": 1, "multilabel": {"threshold": 0.5}}


@pytest.fixture(params=FAMILIES, ids=lambda family: family.model_name)
def trained(request):
    model = request.param(config())
    model.train(records("train"), records("validation"))
    return model


def test_training_learns_separable_topics(trained) -> None:
    assert trained.labels == ["acoustics", "optics"]
    acoustic, optic = trained.predict_batch(["acoustic noise sound", "lazer foton mercek"])
    assert acoustic.label_scores["acoustics"] > acoustic.label_scores["optics"]
    assert optic.label_scores["optics"] > optic.label_scores["acoustics"]


def test_predictions_follow_the_shared_contract(trained) -> None:
    texts = ["  sound   waves ", "laser light", "ses dalgaları"]
    results = trained.predict_batch(iter(texts))
    assert [result.text for result in results] == ["sound waves", "laser light", "ses dalgaları"]
    for result in results:
        assert result.model == trained.model_name
        assert result.language.primary == "und"
        assert set(result.label_scores) == {"acoustics", "optics"}
        assert all(0.0 <= score <= 1.0 for score in result.label_scores.values())
        assert all(item.score >= 0.5 for item in result.labels)
        assert result.metadata["supported_labels"] == ["acoustics", "optics"]
        assert result.metadata["ood_method"] == "max_probability"
        assert 0.0 <= result.metadata["ood_threshold"] <= 1.0
        assert result.latency_ms >= 0
    assert trained.predict("laser").text == "laser"
    assert trained.predict_batch([]) == []


def test_artifact_roundtrip_reproduces_scores(trained, tmp_path) -> None:
    trained.metadata.update({"dataset": "fixture", "track": "en_tr_direct"})
    target = tmp_path / trained.model_name
    trained.save(target)
    loaded = type(trained)()
    loaded.load(target)
    texts = ["sound waves", "lazer ışık", "mixed acoustic laser"]
    for before, after in zip(
        trained.predict_batch(texts), loaded.predict_batch(texts), strict=True
    ):
        assert after.label_scores == pytest.approx(before.label_scores)
        assert after.labels == before.labels
        assert after.metadata["dataset"] == "fixture"
        assert after.metadata["track"] == "en_tr_direct"


def test_load_rejects_tampered_estimator(trained, tmp_path) -> None:
    target = tmp_path / "artifact"
    trained.save(target)
    estimator = target / "estimator.joblib"
    estimator.write_bytes(estimator.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        type(trained)().load(target)


def test_load_rejects_other_family_artifact(trained, tmp_path) -> None:
    target = tmp_path / "artifact"
    trained.save(target)
    other = SVMModel if isinstance(trained, NaiveBayesModel) else NaiveBayesModel
    with pytest.raises(ValueError, match="Incompatible"):
        other().load(target)


@pytest.mark.parametrize("family", FAMILIES)
def test_unfitted_model_refuses_prediction_and_saving(family, tmp_path) -> None:
    model = family()
    with pytest.raises(RuntimeError):
        model.predict("laser")
    with pytest.raises(RuntimeError):
        model.save(tmp_path / "artifact")


@pytest.mark.parametrize("family", FAMILIES)
def test_training_rejects_wrong_split_and_overlap(family) -> None:
    with pytest.raises(ValueError, match="train"):
        family(config()).train(records("validation"))
    train = records("train")
    leaked = [{**train[0], "split": "validation"}]
    with pytest.raises(ValueError, match="id contamination"):
        family(config()).train(train, leaked)


@pytest.mark.parametrize("family", FAMILIES)
def test_training_rejects_validation_translation_of_a_training_pair(family) -> None:
    train, validation = records("train"), records("validation")
    translation = train.pop(1)  # the TR side of train[0]'s project
    assert translation["pair_id"] == train[0]["pair_id"]
    with pytest.raises(ValueError, match="pair_id contamination between train and validation"):
        family(config()).train(train, [*validation, {**translation, "split": "validation"}])
    model = family(config())
    model.train(train, validation)
    assert model.labels == ["acoustics", "optics"]


@pytest.mark.parametrize("family", FAMILIES)
def test_invalid_configuration_is_rejected(family) -> None:
    with pytest.raises(ValueError):
        family({"model": "transformer"})
    with pytest.raises(ValueError):
        family({"multilabel": {"threshold": 1.5}})
    with pytest.raises(ValueError):
        family({"multilabel": {"per_label": {"not_a_label": 0.3}}})


@pytest.mark.parametrize("family", [*FAMILIES, HybridKNNModel])
@pytest.mark.parametrize("value", [True, "0.5", None])
@pytest.mark.parametrize("key", ["threshold", "per_label"])
def test_thresholds_must_be_numbers_not_bools_text_or_null(family, key, value) -> None:
    multilabel = {key: value if key == "threshold" else {"optics": value}}
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        family({"multilabel": multilabel})


@pytest.mark.parametrize("family", [*FAMILIES, HybridKNNModel])
@pytest.mark.parametrize(
    "multilabel",
    [None, {"per_label": None}, {"per_label": {1: 0.3}}, {"per_lable": {"optics": 0.3}}],
)
def test_multilabel_settings_must_be_a_threshold_and_a_label_map(family, multilabel) -> None:
    with pytest.raises(ValueError, match="multilabel"):
        family({"multilabel": multilabel})


def test_svm_needs_enough_positives_for_calibration() -> None:
    train = [row for row in records("train") if "both" not in row["id"]]
    train = [row for row in train if row["labels"] != ["optics"]] + [
        row for row in train if row["labels"] == ["optics"]
    ][:2]
    with pytest.raises(ValueError, match="calibration"):
        SVMModel(config()).train(train)


def test_calibration_splits_never_separate_a_group() -> None:
    from naltra.models.svm.model import calibration_splits

    groups = ["p1", "p1", "p2", "p2", "p3", "p3", "solo"]
    splits = calibration_splits(groups, 3)
    assert len(splits) == 3
    assert sorted(int(row) for _, holdout in splits for row in holdout) == list(range(7))
    for fit_rows, holdout in splits:
        assert not {groups[row] for row in fit_rows} & {groups[row] for row in holdout}


def test_svm_calibrates_only_on_projects_it_did_not_fit(monkeypatch) -> None:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import check_cv

    import naltra.models.svm.model as svm_module

    used = []

    class Recording(CalibratedClassifierCV):
        def fit(self, X, y, **params):
            used.append(list(check_cv(self.cv, y, classifier=True).split(X, y)))
            return super().fit(X, y, **params)

    monkeypatch.setattr(svm_module, "CalibratedClassifierCV", Recording)
    train = records("train")
    SVMModel(config()).train(train)
    groups = [row.get("pair_id") or row["id"] for row in train]
    assert len(used) == 2
    for splits in used:
        for fit_rows, holdout in splits:
            assert not {groups[row] for row in fit_rows} & {groups[row] for row in holdout}


def test_svm_without_translation_pairs_trains_on_a_smoke_subset() -> None:
    from scripts.train_all import smoke_records

    def row(index: int, split: str) -> dict:
        label = "acoustics" if index % 3 == 0 else "optics"
        return {
            "id": f"{split}:r{index:02d}",
            "text": f"{TOPICS[label][0]} {split} item {index}",
            "labels": [label],
            "labels_direct": [label],
            "language": "en",
            "source": "fixture",
            "license": "MIT",
            "split": split,
        }

    # Sole positives r00, r03, r06: ungrouped stratified folds spread them, plain
    # GroupKFold would put all three in one calibration fold.
    track = {
        "train": [row(index, "train") for index in range(9)],
        "validation": [row(index, "validation") for index in range(4)],
        "target_field": "labels_direct",
    }
    train, validation = smoke_records(track, 128, 3)
    model = SVMModel(config())
    model.train(train, validation)
    assert model.labels == ["acoustics", "optics"]


def test_training_cli_registers_classical_models() -> None:
    from scripts.train_all import MODEL_TYPES

    assert MODEL_TYPES["naive_bayes"] is NaiveBayesModel
    assert MODEL_TYPES["svm"] is SVMModel


def test_saved_payload_is_plain_json(trained, tmp_path) -> None:
    target = tmp_path / "artifact"
    trained.save(target)
    payload = json.loads((target / "naltra.json").read_text(encoding="utf-8"))
    assert payload["model"] == trained.model_name
    assert payload["labels"] == ["acoustics", "optics"]
    assert len(payload["estimator_sha256"]) == 64
    assert payload["artifact_version"] == 2
    assert payload["coef_layout"] == "feature_major"


def test_square_label_major_version_1_artifact_is_refused(tmp_path) -> None:
    # Before feature-major storage, version 1 kept coef as labels x features; a square
    # matrix passes the label-count check and would silently score with the wrong axis.
    import joblib
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    from naltra.data.manifest import compute_file_sha256

    model = NaiveBayesModel(config())
    model.train(records("train"))
    target = tmp_path / "legacy"
    model.save(target)
    weights = {"coef": np.diag([3.0, -3.0, 0.0]).astype(np.float32), "intercept": np.zeros(3)}
    vectorizer = TfidfVectorizer().fit(["sound laser photon"])
    joblib.dump({"vectorizer": vectorizer, "weights": weights}, target / "estimator.joblib")
    payload = json.loads((target / "naltra.json").read_text(encoding="utf-8"))
    payload.pop("coef_layout", None)
    payload.update(
        artifact_version=1,
        labels=["acoustics", "optics", "natural_sciences"],
        estimator_sha256=compute_file_sha256(target / "estimator.joblib"),
    )
    (target / "naltra.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="label-major.*retrain"):
        NaiveBayesModel().load(target)


@pytest.mark.parametrize(
    "change",
    [{"artifact_version": 3}, {"coef_layout": "label_major"}, {"coef_layout": None}],
    ids=["unknown_version", "other_layout", "missing_layout"],
)
def test_artifact_without_feature_major_version_2_layout_is_refused(
    trained, tmp_path, change
) -> None:
    target = tmp_path / "artifact"
    trained.save(target)
    payload = json.loads((target / "naltra.json").read_text(encoding="utf-8"))
    payload.update(change)
    payload = {key: value for key, value in payload.items() if value is not None}
    (target / "naltra.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Incompatible"):
        type(trained)().load(target)


@pytest.mark.parametrize("family", FAMILIES)
def test_compact_scores_equal_scikit_learn_one_vs_rest(family) -> None:
    import numpy as np
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer

    model = family(config())
    train = records("train")
    model.train(train)
    texts = [row["text"] for row in train] + ["laser noise", "ses ışık"]
    x = model.vectorizer.transform(texts)
    y = MultiLabelBinarizer(classes=model.labels).fit_transform(
        [row["labels_direct"] for row in train]
    )
    groups = [translation_group(row) for row in train]
    reference = OneVsRestClassifier(model._classifier(groups)).fit(x[: len(train)], y)
    assert np.allclose(model._probabilities(texts), reference.predict_proba(x), atol=1e-5)


def test_turkish_dotted_capital_i_lowercases_like_plain_i() -> None:
    analyzer = NaiveBayesModel()._vectorizer().build_analyzer()
    assert analyzer("İklim değişikliği") == analyzer("iklim değişikliği")


@pytest.mark.parametrize("family", FAMILIES)
def test_label_present_in_every_record_is_rejected(family) -> None:
    train = records("train")
    for row in train:
        row["labels_direct"] = sorted({*row["labels_direct"], "acoustics"})
        row["labels"] = row["labels_direct"]
    with pytest.raises(ValueError, match="every training record"):
        family(config()).train(train)


def test_artifact_from_another_scikit_learn_version_is_rejected(trained, tmp_path) -> None:
    target = tmp_path / "artifact"
    trained.save(target)
    payload = json.loads((target / "naltra.json").read_text(encoding="utf-8"))
    payload["sklearn_version"] = "0.0.1"
    (target / "naltra.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="scikit-learn"):
        type(trained)().load(target)


@pytest.mark.parametrize("family", FAMILIES)
def test_weights_are_stored_feature_major_for_fast_single_text_scoring(family) -> None:
    # Scoring is features @ coef; a label-major matrix would force a full transposed copy
    # of the 473 x 50k weights on every call (measured 37 ms vs 0.01 ms per fold).
    model = family(config())
    model.train(records("train"))
    coef = model.weights["coef"]
    assert coef.shape[-1] == len(model.labels)
    assert coef.flags["C_CONTIGUOUS"]
