"""Jev response adapter implementing the shared prediction contract."""

from naltra.models.external import ExternalServiceModel
from naltra.models.jev.client import JevClient


class JevModel(ExternalServiceModel):
    model_name = "jev"
    client_type = JevClient
