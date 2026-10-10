import json

import httpx
import pytest
from tests.test_kev_model import answers

from naltra.models.kev import KevModel
from naltra.models.kev.client import KevClient


def kev_client(serve, **fields) -> KevClient:
    defaults = {
        "api_key": "secret",
        "base_url": "https://example.test",
        "configured": True,
        "scores_path": "answers",
        "questions": {"acoustics": "Is this project about acoustics?"},
    }
    return KevClient(**(defaults | fields), transport=httpx.MockTransport(serve))


def test_external_contract_mapping_and_secret_free_persistence(tmp_path):
    def serve(request):
        assert request.headers["X-Api-Key"] == "test-secret"
        assert json.loads(request.content)["input"] == "some text"
        assert request.url.path == "/api/classify"
        return httpx.Response(200, json={"result": answers({"Acoustics": 0.8, "Optics": 0.2})})

    client = kev_client(
        serve,
        api_key="test-secret",
        base_url="https://example.test/api",
        request_path="classify",
        text_field="input",
        scores_path="result.answers",
        auth_header="X-Api-Key",
        auth_prefix="",
        questions={"Acoustics": "Is this about sound?", "Optics": "Is this about light?"},
    )
    model = KevModel(
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
    loaded = KevModel()
    loaded.load(tmp_path)
    assert loaded.client is None
    assert loaded.config == model.config
    with pytest.raises(RuntimeError, match="local training"):
        model.train([])
    assert "test-secret" not in repr(client)


@pytest.mark.parametrize(
    "response",
    [
        answers({"acoustics": float("nan")}),
        answers({"acoustics": 1.1}),
        answers({"acoustics": True}),
        {"answers": []},
        {"answers": {}},
        {"wrong": {}},
    ],
)
def test_invalid_service_response(response):
    # JSON cannot encode NaN through httpx's strict response JSON encoder.
    client = kev_client(lambda request: httpx.Response(200, content=json.dumps(response).encode()))
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

    with pytest.raises((RuntimeError, ValueError)) as caught:
        kev_client(serve).predict("valid text")
    assert "secret" not in str(caught.value)


def test_disabled_unconfigured_and_missing_credentials(monkeypatch):
    with pytest.raises(RuntimeError, match="disabled"):
        KevModel(config={"enabled": False}).predict("text")
    with pytest.raises(RuntimeError, match="contract"):
        kev_client(lambda request: httpx.Response(200), configured=False).predict("text")
    monkeypatch.setattr("naltra.models.external.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("KEV_API_KEY", raising=False)
    monkeypatch.delenv("KEV_API_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="KEV_API_KEY"):
        KevClient.from_environment()


@pytest.mark.parametrize(
    "asked,labels,match",
    [
        (["optics"], ["acoustics"], "Unknown"),
        (["acoustics"], ["optics", "acoustics"], "every supported"),
    ],
)
def test_external_label_space_validation(asked, labels, match):
    client = kev_client(
        lambda request: httpx.Response(200, json=answers({label: 0.5 for label in asked})),
        questions={label: f"Is this project about {label}?" for label in asked},
    )
    model = KevModel(client, {"enabled": True, "supported_labels": labels})
    with pytest.raises(ValueError, match=match):
        model.predict("valid text")


def test_service_configuration_rejects_inline_credentials():
    with pytest.raises(ValueError, match="environment"):
        KevModel(config={"api_key": "should-not-be-saved"})
    with pytest.raises(ValueError, match="timeout"):
        KevModel(config={"request_timeout_seconds": -1})


def test_api_key_is_not_sent_over_remote_plain_http():
    def serve(request):
        raise AssertionError("The API key left over plain http.")

    with pytest.raises(ValueError, match="https"):
        kev_client(serve, base_url="http://api.example.com").predict("valid text")


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "[::1]"])
def test_plain_http_is_allowed_for_a_loopback_server(host):
    client = kev_client(
        lambda request: httpx.Response(200, json=answers({"acoustics": 0.5})),
        base_url=f"http://{host}:8009",
    )
    assert client.predict("valid text") == {"scores": {"acoustics": 0.5}}


CONTRACT_STRINGS = ("request_path", "text_field", "scores_path", "auth_header", "auth_prefix")


@pytest.mark.parametrize(
    "config,match",
    [
        ({"request_timeout_seconds": "30"}, "timeout"),
        ({"request_timeout_seconds": 10**400}, "timeout"),
        ({"label_map": ["x"]}, "label_map"),
        ({"label_map": {1: "acoustics"}}, "label_map"),
        ({"label_map": {"Acoustics": 1}}, "label_map"),
        ({"multilabel": ["x"]}, "multilabel"),
        ({"multilabel": {"threshold": "0.5"}}, "threshold"),
        ({"multilabel": {"threshold": None}}, "threshold"),
        ({"multilabel": {"threshold": True}}, "threshold"),
        ({"multilabel": {"threshold": float("nan")}}, "threshold"),
        ({"multilabel": {"threshold": 1.5}}, "threshold"),
        ({"multilabel": {"threshold": 10**400}}, "threshold"),
        ({"multilabel": {"per_label": ["acoustics"]}}, "per_label"),
        ({"multilabel": {"per_label": None}}, "per_label"),
        ({"multilabel": {"per_label": {"acoustics": "0.4"}}}, "per_label"),
        ({"multilabel": {"per_label": {"acoustics": True}}}, "per_label"),
        ({"multilabel": {"per_label": {"acoustics": -0.1}}}, "per_label"),
        ({"multilabel": {"per_label": {"not_a_label": 0.4}}}, "per_label"),
        ({"multilabel": {"per_lable": {"acoustics": 0.7}}}, "multilabel"),
        ({"label_map": {"Acoustics": "acoustics", "Sound": "acoustics"}}, "label_map"),
        ({"contract": None}, "contract"),
        *(({"contract": {name: 5}}, "contract") for name in CONTRACT_STRINGS),
    ],
)
def test_wrongly_typed_service_configuration_raises_value_error(config, match):
    with pytest.raises(ValueError, match=match):
        KevModel(config=config)


def test_per_label_thresholds_for_supported_labels_are_accepted():
    model = KevModel(config={"multilabel": {"threshold": 0.4, "per_label": {"acoustics": 0.7}}})
    assert model.config["multilabel"] == {"threshold": 0.4, "per_label": {"acoustics": 0.7}}


def without(record: dict, field: str) -> dict:
    return {key: value for key, value in record.items() if key != field}


@pytest.mark.parametrize(
    "edit",
    [
        lambda payload: [payload],
        lambda payload: without(payload, "model"),
        lambda payload: without(payload, "taxonomy"),
        lambda payload: payload | {"config": None},
        lambda payload: payload | {"config": without(payload["config"], "multilabel")},
        lambda payload: payload | {"config": without(payload["config"], "model")},
    ],
    ids=["list", "no_model", "no_taxonomy", "null_config", "no_multilabel", "no_config_model"],
)
def test_malformed_service_artifact_raises_value_error(tmp_path, edit):
    KevModel().save(tmp_path)
    path = tmp_path / "naltra.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(edit(payload)), encoding="utf-8")
    with pytest.raises(ValueError):
        KevModel().load(tmp_path)


@pytest.mark.parametrize(
    "template",
    [
        "{{name}}",
        "Is this project about {field}?",
        "{name} {field}",
        "{0} {name}",
        "{} {name}",
        "{name.upper}",
        "{name[0]}",
        "{name!r}",
        "{name:.0}",
        "{name",
        "about {name}}",
        5,
        None,
    ],
)
def test_question_template_must_insert_the_label_name_and_nothing_else(template):
    with pytest.raises(ValueError, match="question_template"):
        KevModel(config={"question_template": template})


@pytest.mark.parametrize("template", ["{name}", "Does {{this}} cover {name}?"])
def test_question_template_gives_each_label_its_own_question(template):
    questions = KevModel(config={"question_template": template}).questions()
    assert questions["acoustics"] != questions["optics"]
    assert "acoustics" in questions["acoustics"]
