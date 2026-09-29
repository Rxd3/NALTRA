"""Language detection boundary used by the prediction pipeline."""

from __future__ import annotations

from typing import Protocol

from naltra.schemas.prediction import LanguageInfo


class LanguageDetector(Protocol):
    """Protocol for a future English/Turkish language detector."""

    def detect(self, text: str) -> LanguageInfo:
        """Detect the primary language and code-switching state."""
        ...


class UndeterminedLanguageDetector:
    """Safe placeholder used until a detector is selected and validated."""

    def detect(self, text: str) -> LanguageInfo:
        if not text.strip():
            raise ValueError("Cannot detect the language of empty text.")
        return LanguageInfo(primary="und", is_code_switched=False)
