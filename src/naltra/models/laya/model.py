"""Laya response adapter implementing the shared prediction contract."""

from naltra.models.external import ExternalServiceModel
from naltra.models.laya.client import LayaClient


class LayaModel(ExternalServiceModel):
    model_name = "laya"
    client_type = LayaClient
