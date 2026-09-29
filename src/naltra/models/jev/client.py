"""Configuration-only Jev client placeholder; no network requests are made."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class JevClient:
    """Future client boundary for the external Jev service."""

    api_key: str
    base_url: str

    @classmethod
    def from_environment(cls) -> JevClient:
        api_key = os.getenv("JEV_API_KEY", "")
        base_url = os.getenv("JEV_API_BASE_URL", "")
        if not api_key or not base_url:
            raise RuntimeError("JEV_API_KEY and JEV_API_BASE_URL must be set in the environment.")
        return cls(api_key=api_key, base_url=base_url)

    def predict(self, text: str) -> dict[str, object]:
        del text
        raise NotImplementedError("Jev network integration has not been implemented.")
