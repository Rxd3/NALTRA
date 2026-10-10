"""Jev as a zero-shot ensemble member, asked exactly the questions Kev is asked."""

from __future__ import annotations

from naltra.models.jev.client import JevClient
from naltra.models.kev.model import KevModel


class JevModel(KevModel):
    model_name = "jev"
    client_type = JevClient
