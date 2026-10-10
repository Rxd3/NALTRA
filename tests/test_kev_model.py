"""Kev (open System One decision model) adapter: one yes/no question per label."""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from naltra.models.kev import KevModel
from naltra.models.kev.client import KevClient

QUESTIONS = {
    "acoustics": "Is this project about acoustics?",
    "optics": "Is this project about optics?",
}


def client(serve) -> KevClient:
    return KevClient(
        api_key="local",
        base_url="http://localhost:8009",
        configured=True,
        request_path="v1/systemone",
        text_field="state",
        scores_path="answers",
        questions=QUESTIONS,
        transport=httpx.MockTransport(serve),
    )


def answers(scores: dict) -> dict:
    return {"answers": {key: {"type": "noul", "noul": value} for key, value in scores.items()}}


def test_client_asks_one_noul_per_label_and_reads_p_yes() -> None:
    seen = []

    def serve(request):
        seen.append(json.loads(request.content))
        assert request.url.path == "/v1/systemone"
        return httpx.Response(200, json=answers({"acoustics": 0.9, "optics": 0.1}))

    assert client(serve).predict("  Sound  waves ") == {"scores": {"acoustics": 0.9, "optics": 0.1}}
    body = seen[0]
    assert body["state"] == "Sound waves"
    assert body["model"] == "kev-latest"
    assert body["questions"] == {
        label: {"type": "noul", "instructions": text} for label, text in QUESTIONS.items()
    }


@pytest.mark.parametrize(
    "payload",
    [
        answers({"acoustics": 0.9}),
        answers({"acoustics": 0.9, "optics": 1.5}),
        {"answers": {"acoustics": {"type": "choice", "choice": "x"}, "optics": {"noul": 0.2}}},
        {"no_answers": {}},
    ],
)
def test_client_rejects_incomplete_or_malformed_answers(payload) -> None:
    with pytest.raises(ValueError):
        client(lambda request: httpx.Response(200, json=payload)).predict("text")


@pytest.mark.parametrize("answer", [None, 0.5, "yes", [0.5]])
def test_client_rejects_non_object_answers_with_value_error(answer) -> None:
    payload = {"answers": {label: answer for label in QUESTIONS}}
    with pytest.raises(ValueError, match="noul"):
        client(lambda request: httpx.Response(200, json=payload)).predict("text")


def test_model_defaults_to_every_direct_label_with_english_questions() -> None:
    model = KevModel()
    taxonomy = json.loads(open("taxonomy/taxonomy.json", encoding="utf-8").read())
    direct = sorted(x["id"] for x in taxonomy["labels"] if x.get("is_direct_supported"))
    assert model.labels == direct
    questions = model.questions()
    assert len(questions) == len(direct) == 473
    assert questions["acoustics"] == "Is this project about acoustics?"


def test_model_scores_through_an_injected_client() -> None:
    def serve(request):
        asked = json.loads(request.content)["questions"]
        return httpx.Response(200, json=answers({label: 0.6 for label in asked}))

    labels = ["acoustics", "optics"]
    model = KevModel(
        client(serve),
        {"enabled": True, "supported_labels": labels, "multilabel": {"threshold": 0.5}},
    )
    result = model.predict("Sound waves and lasers")
    assert result.label_scores == {"acoustics": 0.6, "optics": 0.6}
    assert model.labels == labels
    assert np.isclose(sum(item.score for item in result.labels), 1.2)
