"""Language detection boundary used by the prediction pipeline."""

from __future__ import annotations

import re
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


class LinguaLanguageDetector:
    """Offline EN/TR detector; mixed-language decisions are experimental."""

    def __init__(self, minimum_confidence: float = 0.6, minimum_share: float = 0.2) -> None:
        if not 0.5 <= minimum_confidence <= 1 or not 0 < minimum_share <= 0.5:
            raise ValueError("Invalid language detector thresholds.")
        self.minimum_confidence = minimum_confidence
        self.minimum_share = minimum_share
        self._detector = None

    def detect(self, text: str) -> LanguageInfo:
        from lingua import Language, LanguageDetectorBuilder

        if not text.strip():
            raise ValueError("Cannot detect the language of empty text.")
        if len(re.findall(r"[^\W\d_]+", text)) < 2:
            return LanguageInfo(primary="und")
        if self._detector is None:
            self._detector = LanguageDetectorBuilder.from_languages(
                Language.ENGLISH, Language.TURKISH
            ).build()
        confidence = self._detector.compute_language_confidence_values(text)
        codes = {Language.ENGLISH: "en", Language.TURKISH: "tr"}
        counts = {"en": 0, "tr": 0}
        for section in self._detector.detect_multiple_languages_of(text):
            fragment = text[section.start_index : section.end_index]
            words = len(re.findall(r"[^\W\d_]+", fragment))
            section_confidence = self._detector.compute_language_confidence(
                fragment, section.language
            )
            if words >= 2 and section_confidence >= self.minimum_confidence:
                counts[codes[section.language]] += words
        total = sum(counts.values())
        if total and all(
            count >= 2 and count / total >= self.minimum_share for count in counts.values()
        ):
            return LanguageInfo(primary="en-tr", is_code_switched=True)
        if not confidence or confidence[0].value < self.minimum_confidence:
            return LanguageInfo(primary="und")
        return LanguageInfo(primary=codes[confidence[0].language])
