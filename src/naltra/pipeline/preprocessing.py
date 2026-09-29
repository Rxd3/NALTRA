"""Shared inference-time text preprocessing."""

import re


def normalize_text(text: str) -> str:
    """Trim text and collapse repeated whitespace without changing its language."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return re.sub(r"\s+", " ", text).strip()
