"""Dataset preprocessing helpers."""

from naltra.pipeline.preprocessing import normalize_text


def preprocess_record_text(text: str) -> str:
    """Apply the shared, language-preserving text normalization."""
    return normalize_text(text)
