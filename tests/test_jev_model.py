"""Jev (TypeSafe's hosted System One model) shares Kev's question protocol."""

from __future__ import annotations

import json

import httpx
import pytest
from scripts import predict_all

from naltra.models.jev import JevModel
from naltra.models.jev.client import JevClient
from naltra.models.kev import KevModel


def test_client_asks_the_hosted_jev_alias_at_the_system_one_path() -> None:
    seen = []

    def serve(request):
        seen.append((request.url.path, json.loads(request.content)))
        asked = seen[-1][1]["questions"]
        return httpx.Response(
            200, json={"answers": {k: {"type": "noul", "noul": 0.7} for k in asked}}
        )

    client = JevClient(
        api_key="key",
        base_url="https://api.example",
        configured=True,
        request_path="v1/systemone",
        text_field="state",
        scores_path="answers",
        questions={"optics": "Is this project about optics?"},
        transport=httpx.MockTransport(serve),
    )
    assert client.predict("Lasers") == {"scores": {"optics": 0.7}}
    path, body = seen[0]
    assert path == "/v1/systemone"
    assert body["model"] == "jev-latest" and body["state"] == "Lasers"


def test_model_is_enabled_and_asks_the_same_questions_as_kev() -> None:
    jev, kev = JevModel(), KevModel()
    assert jev.config["enabled"] is True
    assert jev.config["contract"]["request_path"] == "v1/systemone"
    assert jev.labels == kev.labels and len(jev.labels) == 473
    assert jev.questions() == kev.questions()


def test_model_builds_a_jev_client_from_its_own_environment(monkeypatch) -> None:
    monkeypatch.setenv("JEV_API_KEY", "secret")
    monkeypatch.setenv("JEV_API_BASE_URL", "https://api.example")
    client = JevModel()._make_client()
    assert type(client) is JevClient
    assert client.model_alias == "jev-latest" and client.base_url == "https://api.example"
    assert len(client.questions) == 473


def test_missing_credentials_are_refused(monkeypatch) -> None:
    monkeypatch.setattr("naltra.models.external.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.delenv("JEV_API_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="JEV_API_KEY"):
        JevModel()._make_client()


def test_predict_all_can_score_jev() -> None:
    assert "jev" in predict_all.SERVICE_TYPES
