"""Hybrid sparse+dense retrieval kNN member (RRF fusion), with an offline fake encoder."""

from __future__ import annotations

import hashlib
import json
import re
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from tests.test_classical_models import records

from naltra.data.manifest import compute_file_sha256
from naltra.models.classical import ESTIMATOR_FILE
from naltra.models.hybrid_knn import HybridKNNModel
from naltra.models.hybrid_knn.model import reciprocal_rank_fusion


def word_bucket(word: str) -> int:
    return int(hashlib.md5(word.encode()).hexdigest(), 16) % 32


def fake_encoder(texts):
    """Deterministic hashed bag-of-words embedding, L2-normalised like e5."""
    vectors = np.zeros((len(texts), 32), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in text.lower().split():
            vectors[row, word_bucket(word)] += 1
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)


def fake_pretrained(monkeypatch, *, reversed_axes: bool = False) -> list:
    """Replace only the Hugging Face loaders, so the real E5 pooling path runs offline."""
    import torch
    import transformers

    def tokenizer(texts, max_length, **_):
        ids = [[word_bucket(word) for word in text.lower().split()][:max_length] for text in texts]
        width = max(map(len, ids))
        return transformers.BatchEncoding(
            {
                "input_ids": torch.tensor([row + [0] * (width - len(row)) for row in ids]),
                "attention_mask": torch.tensor(
                    [[1] * len(row) + [0] * (width - len(row)) for row in ids]
                ),
            }
        )

    def network(input_ids, attention_mask):
        hidden = torch.nn.functional.one_hot(input_ids, 32).float()
        return SimpleNamespace(last_hidden_state=hidden.flip(-1) if reversed_axes else hidden)

    network.to = lambda device: network
    network.eval = lambda: network
    calls = []

    def loader(kind, loaded):
        def from_pretrained(name, **options):
            calls.append((kind, name, options))
            return loaded

        return from_pretrained

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", loader("tokenizer", tokenizer)
    )
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", loader("model", network))
    return calls


def config(**retrieval) -> dict:
    return {
        "target_field": "labels_direct",
        "retrieval": {"neighbors": 3, **retrieval},
        "multilabel": {"threshold": 0.5},
    }


@pytest.fixture
def trained():
    model = HybridKNNModel(config(), encoder=fake_encoder)
    model.train(records("train"), records("validation"))
    return model


def test_rrf_rewards_documents_ranked_high_by_both_retrievers() -> None:
    sparse = np.array([[0.9, 0.5, 0.1]])
    dense = np.array([[0.2, 0.8, 0.7]])
    fused = reciprocal_rank_fusion([sparse, dense], neighbors=2, k=60)
    # Doc 1 is 2nd and 1st; doc 0 is 1st only; doc 2 is 2nd only.
    assert fused[0, 1] > fused[0, 0] > fused[0, 2] > 0
    assert fused[0, 1] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_gives_no_weight_to_zero_similarity_documents() -> None:
    sparse = np.array([[0.0, 0.4, 0.0], [0.0, 0.0, 0.0]])
    fused = reciprocal_rank_fusion([sparse], neighbors=3, k=60)
    assert fused[0] == pytest.approx([0.0, 1 / 61, 0.0])
    assert not fused[1].any()


def test_query_without_shared_vocabulary_abstains() -> None:
    train = records("train")
    scores = []
    for rows in (train, train[::-1]):
        model = HybridKNNModel(config(dense=False), encoder=fake_encoder)
        model.train(rows)
        scores.append(model.predict("xylophone quartz").label_scores)
    assert scores[0] == scores[1] == {"acoustics": 0.0, "optics": 0.0}


def test_load_uses_callers_device_and_artifact_encoder_settings(tmp_path) -> None:
    producer = HybridKNNModel(
        {**config(), "device": "cuda", "encoder": {"pretrained_name": "fixture/e5"}},
        encoder=fake_encoder,
    )
    producer.train(records("train"))
    producer.save(tmp_path / "knn")
    loaded = HybridKNNModel({"device": "cpu"})
    stale = loaded.encoder
    loaded.load(tmp_path / "knn")
    assert loaded.config["device"] == "cpu"
    assert loaded.encoder is not stale
    assert loaded.encoder.settings["device"] == "cpu"
    assert loaded.encoder.settings["pretrained_name"] == "fixture/e5"
    unpinned = HybridKNNModel()
    unpinned.load(tmp_path / "knn")
    assert unpinned.encoder.settings["device"] == "auto"


def test_neighbour_labels_vote_into_probabilities(trained) -> None:
    acoustic, optic = trained.predict_batch(["sound waves acoustic noise", "lazer ışık mercek"])
    assert acoustic.label_scores["acoustics"] > acoustic.label_scores["optics"]
    assert optic.label_scores["optics"] > optic.label_scores["acoustics"]
    for result in (acoustic, optic):
        assert all(0 <= v <= 1 for v in result.label_scores.values())
        assert result.metadata["calibration"] == "none"


def test_each_retriever_can_run_alone() -> None:
    for retrieval in ({"dense": False}, {"sparse": False}):
        model = HybridKNNModel(config(**retrieval), encoder=fake_encoder)
        model.train(records("train"))
        result = model.predict("laser light lens photon")
        assert result.label_scores["optics"] > result.label_scores["acoustics"]
    with pytest.raises(ValueError, match="retriever"):
        HybridKNNModel(config(sparse=False, dense=False))


def test_roundtrip_keeps_index_and_scores(trained, tmp_path) -> None:
    trained.save(tmp_path / "knn")
    loaded = HybridKNNModel(encoder=fake_encoder)
    loaded.load(tmp_path / "knn")
    texts = ["sound waves", "lazer ışık"]
    for before, after in zip(
        trained.predict_batch(texts), loaded.predict_batch(texts), strict=True
    ):
        assert after.label_scores == pytest.approx(before.label_scores)


def test_invalid_retrieval_settings_are_rejected() -> None:
    with pytest.raises(ValueError):
        HybridKNNModel(config(neighbors=0))
    with pytest.raises(ValueError):
        HybridKNNModel(config(rrf_k=-1))


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
@pytest.mark.parametrize("key", ["sparse", "dense"])
def test_retriever_switches_must_be_booleans(key, value) -> None:
    with pytest.raises(ValueError, match=f"retrieval.{key}"):
        HybridKNNModel(config(**{key: value}))


def test_sparse_only_retrieval_needs_no_encoder_settings() -> None:
    HybridKNNModel({**config(dense=False), "encoder": None})


def test_reload_refuses_a_query_encoder_that_drifted_from_the_artifact(
    tmp_path, monkeypatch
) -> None:
    fake_pretrained(monkeypatch)
    producer = HybridKNNModel({**config(sparse=False), "device": "cpu"})
    producer.train(records("train"))
    producer.save(tmp_path / "knn")
    query = "acoustic sound waves"
    same = HybridKNNModel({"device": "cpu"})
    same.load(tmp_path / "knn")
    assert same.predict(query).label_scores == producer.predict(query).label_scores
    # A changed snapshot: same name, embedding axes reversed.
    fake_pretrained(monkeypatch, reversed_axes=True)
    drifted = HybridKNNModel({"device": "cpu"})
    drifted.load(tmp_path / "knn")
    with pytest.raises(ValueError, match="intfloat/multilingual-e5-base"):
        drifted.predict(query)


def test_both_pretrained_loads_request_the_pinned_revision(monkeypatch) -> None:
    calls = fake_pretrained(monkeypatch)
    model = HybridKNNModel({**config(sparse=False), "device": "cpu"})
    model.train(records("train"))
    revision = model.config["encoder"]["revision"]
    assert re.fullmatch(r"[0-9a-f]{40}", str(revision))
    assert calls == [
        (kind, "intfloat/multilingual-e5-base", {"revision": revision})
        for kind in ("tokenizer", "model")
    ]


@pytest.mark.parametrize(
    "revision", [None, "main", "v1.0.0", "d128750", "D128750597153BB5987E10B1C3493A34E5A4502A"]
)
def test_dense_config_refuses_a_revision_that_is_not_a_full_commit_sha(
    monkeypatch, revision
) -> None:
    calls = fake_pretrained(monkeypatch)
    unpinned = {"encoder": {"revision": revision}}
    # Refused on construction, before any TF-IDF fitting or encoder download.
    with pytest.raises(ValueError, match="encoder.revision"):
        HybridKNNModel({**config(), **unpinned, "device": "cpu"})
    assert calls == []
    HybridKNNModel({**config(dense=False), **unpinned}).train(records("train"))


def test_load_refuses_a_dense_artifact_config_with_an_unpinned_revision(trained, tmp_path) -> None:
    trained.save(tmp_path / "knn")
    path = tmp_path / "knn" / "naltra.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["config"]["encoder"]["revision"] = "main"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="encoder.revision"):
        HybridKNNModel(encoder=fake_encoder).load(tmp_path / "knn")


def test_dense_training_accepts_the_shipped_commit_pin() -> None:
    sha = "d128750597153bb5987e10b1c3493a34e5a4502a"
    model = HybridKNNModel({**config(), "encoder": {"revision": sha}}, encoder=fake_encoder)
    model.train(records("train"))
    assert model.weights["fingerprint"]["revision"] == sha


@pytest.mark.parametrize(
    "tamper",
    [
        lambda weights: weights.pop("fingerprint"),
        lambda weights: weights.update(fingerprint=["intfloat/multilingual-e5-base"]),
        lambda weights: weights["fingerprint"].pop("revision"),
        lambda weights: weights["fingerprint"].pop("canary"),
    ],
    ids=["missing", "not_a_dict", "no_revision", "no_canary"],
)
def test_dense_artifact_without_a_wellformed_fingerprint_is_refused(
    trained, tmp_path, tamper
) -> None:
    trained.save(tmp_path / "knn")
    estimator = tmp_path / "knn" / ESTIMATOR_FILE
    state = joblib.load(estimator)
    tamper(state["weights"])
    joblib.dump(state, estimator)
    payload = json.loads((tmp_path / "knn" / "naltra.json").read_text(encoding="utf-8"))
    payload["estimator_sha256"] = compute_file_sha256(estimator)
    (tmp_path / "knn" / "naltra.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="retrain"):
        HybridKNNModel(encoder=fake_encoder).load(tmp_path / "knn")


@pytest.mark.parametrize(
    "key,value", [("pretrained_name", "other"), ("revision", "0123456789abcdef" * 2 + "0" * 8)]
)
def test_load_refuses_a_config_encoder_other_than_the_fingerprinted_one(
    trained, tmp_path, key, value
) -> None:
    trained.save(tmp_path / "knn")
    path = tmp_path / "knn" / "naltra.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["config"]["encoder"][key] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        HybridKNNModel(encoder=fake_encoder).load(tmp_path / "knn")


def test_sparse_only_artifact_needs_no_encoder_fingerprint(tmp_path) -> None:
    sparse = HybridKNNModel(config(dense=False))
    sparse.train(records("train"))
    sparse.save(tmp_path / "sparse")
    loaded = HybridKNNModel()
    loaded.load(tmp_path / "sparse")
    text = "laser light lens photon"
    assert loaded.predict(text).label_scores == sparse.predict(text).label_scores


@pytest.mark.parametrize(
    "dense,switch",
    [(True, {"dense": False}), (True, {"sparse": False}), (False, {"dense": True})],
    ids=["hybrid_as_sparse", "hybrid_as_dense", "sparse_as_hybrid"],
)
def test_load_refuses_a_retrieval_config_other_than_the_saved_index(
    tmp_path, dense, switch
) -> None:
    producer = HybridKNNModel(config(dense=dense), encoder=fake_encoder)
    producer.train(records("train"))
    producer.save(tmp_path / "knn")
    path = tmp_path / "knn" / "naltra.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["config"]["retrieval"].update(switch)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="index and retrieval config disagree; retrain"):
        HybridKNNModel(encoder=fake_encoder).load(tmp_path / "knn")
