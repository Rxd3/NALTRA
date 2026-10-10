"""System One client that asks one yes/no ("noul") question per label in a single request."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from naltra.models.external import JSONServiceClient, _is_probability
from naltra.pipeline.preprocessing import normalize_text


@dataclass(slots=True)
class KevClient(JSONServiceClient):
    questions: dict[str, str] = field(default_factory=dict)
    model_alias: str = "kev-latest"
    service_name = "kev"

    def predict(self, text: str) -> dict[str, Any]:
        text = normalize_text(text)
        if not text:
            raise ValueError("Service prediction text cannot be empty.")
        if not self.questions:
            raise RuntimeError("Kev needs one question per supported label.")
        body = {
            self.text_field: text,
            "model": self.model_alias,
            "questions": {
                label: {"type": "noul", "instructions": question}
                for label, question in self.questions.items()
            },
        }
        answers = self._at_scores_path(self._post(body))
        if not isinstance(answers, dict) or set(answers) != set(self.questions):
            raise ValueError("Kev must answer every question exactly once.")
        scores = {}
        for label, answer in answers.items():
            if (
                not isinstance(answer, dict)
                or answer.get("type") != "noul"
                or not _is_probability(answer.get("noul"))
            ):
                raise ValueError("Kev answers must be noul probabilities in [0, 1].")
            scores[label] = float(answer["noul"])
        return {"scores": scores}
