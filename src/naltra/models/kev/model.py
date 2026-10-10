"""Kev as a zero-shot ensemble member: p(yes) for "Is this project about <label>?"."""

from __future__ import annotations

import json
from collections.abc import Mapping
from string import Formatter
from typing import Any

from naltra.data.manifest import REPO_ROOT
from naltra.models.external import ExternalServiceModel
from naltra.models.kev.client import KevClient


def _taxonomy() -> list[dict[str, Any]]:
    path = REPO_ROOT / "taxonomy/taxonomy.json"
    return json.loads(path.read_text(encoding="utf-8"))["labels"]


class KevModel(ExternalServiceModel):
    """Not fine-tuned on CORDIS labels; questions stay English for EN, TR and code-switch inputs."""

    model_name = "kev"
    client_type = KevClient
    extra_config_keys = frozenset({"question_template"})

    def __init__(
        self, client: KevClient | None = None, config: Mapping[str, Any] | None = None
    ) -> None:
        direct = sorted(item["id"] for item in _taxonomy() if item.get("is_direct_supported"))
        config = dict(config or {})
        if not config.get("supported_labels"):
            config["supported_labels"] = direct
        super().__init__(client, config)

    @property
    def labels(self) -> list[str]:
        return list(self.config["supported_labels"])

    def questions(self) -> dict[str, str]:
        names = {item["id"]: item["name"] for item in _taxonomy()}
        template = self.config["question_template"]
        return {label: template.format(name=names[label]) for label in self.labels}

    def _validate_config(self) -> None:
        super()._validate_config()
        template = self.config.get("question_template")
        try:
            fields = [item[1:] for item in Formatter().parse(template) if item[1] is not None]
        except (TypeError, ValueError):  # not a string, or unbalanced braces
            fields = []
        # Only plain {name} fields, so every label gets its own question.
        if not fields or any(field != ("name", "", None) for field in fields):
            raise ValueError(
                "question_template must contain a {name} placeholder and no other fields."
            )

    def _make_client(self) -> KevClient:
        client = self.client_type.from_environment(self.config)
        client.questions = self.questions()
        return client
