"""Configurable Laya JSON service client; requires an explicit provider contract."""

from naltra.models.external import JSONServiceClient


class LayaClient(JSONServiceClient):
    service_name = "laya"
