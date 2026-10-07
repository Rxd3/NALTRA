"""Configurable Jev JSON service client; requires an explicit provider contract."""

from naltra.models.external import JSONServiceClient


class JevClient(JSONServiceClient):
    service_name = "jev"
