import json

import httpx
import pytest

from naltra.models.jev import JevModel
from naltra.models.jev.client import JevClient
from naltra.models.laya import LayaModel
from naltra.models.laya.client import LayaClient


@pytest.mark.parametrize("model_type,client_type", [(JevModel, JevClient), (LayaModel, LayaClient)])
def test_external_contract_mapping_and_secret_free_persistence(model_type, client_type, tmp_path):
    requests = []

    def serve(request):
        requests.append(request)
        assert request.headers["X-Api-Key"] == "test-secret"
        assert json.loads(request.content) == {"input": "some text"}
        assert request.url.path == "/api/classify"
        return httpx.Response(
            200, json={"result": {"probabilities": {"Sports": 0.8, "Health": 0.2}}}
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
            "supported_labels": ["health", "sport"],
            "label_map": {"Sports": "sport", "Health": "health"},
        },
    )
    result = model.predict(" some   text ")
    assert result.label_scores == {"sport": 0.8, "health": 0.2}
    assert [label.label for label in result.labels] == ["sport"]
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
        {"scores": {"sport": float("nan")}},
        {"scores": {"sport": 1.1}},
        {"scores": {"sport": True}},
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
    "scores,match", [({"health": 0.5}, "Unknown"), ({"sport": 0.5}, "every supported")]
)
def test_external_label_space_validation(scores, match):
    client = JevClient(
        "secret",
        "https://example.test",
        configured=True,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"scores": scores})),
    )
    labels = ["sport"] if match == "Unknown" else ["health", "sport"]
    model = JevModel(client, {"enabled": True, "supported_labels": labels})
    with pytest.raises(ValueError, match=match):
        model.predict("valid text")


def test_service_configuration_rejects_inline_credentials():
    with pytest.raises(ValueError, match="environment"):
        JevModel(config={"api_key": "should-not-be-saved"})
    with pytest.raises(ValueError, match="timeout"):
        JevModel(config={"request_timeout_seconds": -1})
