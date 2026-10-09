import json

import httpx
import pytest

from naltra.models.jev import JevModel
from naltra.models.jev.client import JevClient
from naltra.models.laya import LayaModel
from naltra.models.laya.client import LayaClient


@pytest.mark.parametrize("model_type,client_type", [(JevModel, JevClient)])
def test_external_contract_mapping_and_secret_free_persistence(model_type, client_type, tmp_path):
    requests = []

    def serve(request):
        requests.append(request)
        assert request.headers["X-Api-Key"] == "test-secret"
        assert json.loads(request.content) == {"input": "some text"}
        assert request.url.path == "/api/classify"
        return httpx.Response(
            200, json={"result": {"probabilities": {"Acoustics": 0.8, "Optics": 0.2}}}
        )

    client = client_type(
        api_key="test-secret",
        base_url="https://example.test/api",
        configured=True,
        request_path="classify",
        text_field="input",
        scores_path="result.probabilities",
        auth_header="X-Api-Key",
        auth_prefix="",
        transport=httpx.MockTransport(serve),
    )
    model = model_type(
        client,
        {
            "enabled": True,
            "supported_labels": ["optics", "acoustics"],
            "label_map": {"Acoustics": "acoustics", "Optics": "optics"},
        },
    )
    result = model.predict(" some   text ")
    assert result.label_scores == {"acoustics": 0.8, "optics": 0.2}
    assert [label.label for label in result.labels] == ["acoustics"]
    assert len(model.predict_batch(["some text", "some text"])) == 2
    model.save(tmp_path)
    serialized = (tmp_path / "naltra.json").read_text()
    assert "test-secret" not in serialized
    assert "example.test" not in serialized
    loaded = model_type()
    loaded.load(tmp_path)
    assert loaded.client is None
    assert loaded.config == model.config
    with pytest.raises(RuntimeError, match="local training"):
        model.train([])
    assert "test-secret" not in repr(client)


@pytest.mark.parametrize(
    "response",
    [
        {"scores": {"acoustics": float("nan")}},
        {"scores": {"acoustics": 1.1}},
        {"scores": {"acoustics": True}},
        {"scores": []},
        {"scores": {}},
        {"wrong": {}},
    ],
)
def test_invalid_service_response(response):
    # JSON cannot encode NaN through httpx's strict response JSON encoder.
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=json.dumps(response).encode())
    )
    client = JevClient("secret", "https://example.test", configured=True, transport=transport)
    with pytest.raises(ValueError):
        client.predict("valid text")


@pytest.mark.parametrize("error", ["timeout", "http", "json"])
def test_service_errors_are_actionable_and_do_not_expose_secrets(error):
    def serve(request):
        if error == "timeout":
            raise httpx.ReadTimeout("secret should not be logged", request=request)
        if error == "http":
            return httpx.Response(401)
        return httpx.Response(200, content=b"not JSON")

    client = JevClient(
        "secret", "https://example.test", configured=True, transport=httpx.MockTransport(serve)
    )
    with pytest.raises((RuntimeError, ValueError)) as caught:
        client.predict("valid text")
    assert "secret" not in str(caught.value)


def test_disabled_unconfigured_and_missing_credentials(monkeypatch):
    with pytest.raises(RuntimeError, match="disabled"):
        JevModel().predict("text")
    with pytest.raises(RuntimeError, match="contract"):
        JevClient("secret", "https://example.test").predict("text")
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.delenv("JEV_API_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="JEV_API_KEY"):
        JevClient.from_environment()


@pytest.mark.parametrize(
    "scores,match", [({"optics": 0.5}, "Unknown"), ({"acoustics": 0.5}, "every supported")]
)
def test_external_label_space_validation(scores, match):
    client = JevClient(
        "secret",
        "https://example.test",
        configured=True,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"scores": scores})),
    )
    labels = ["acoustics"] if match == "Unknown" else ["optics", "acoustics"]
    model = JevModel(client, {"enabled": True, "supported_labels": labels})
    with pytest.raises(ValueError, match=match):
        model.predict("valid text")


def test_service_configuration_rejects_inline_credentials():
    with pytest.raises(ValueError, match="environment"):
        JevModel(config={"api_key": "should-not-be-saved"})
    with pytest.raises(ValueError, match="timeout"):
        JevModel(config={"request_timeout_seconds": -1})


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
