"""Jev, TypeSafe's hosted System One model: Kev's protocol under the jev-latest alias."""

from __future__ import annotations

from dataclasses import dataclass

from naltra.models.kev.client import KevClient


@dataclass(slots=True)
class JevClient(KevClient):
    model_alias: str = "jev-latest"
    service_name = "jev"
