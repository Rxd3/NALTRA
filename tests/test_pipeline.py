from collections.abc import Iterable
from pathlib import Path
from typing import Any

from naltra.models.base import BaseNALTRAModel
from naltra.pipeline.prediction import PredictionPipeline
from naltra.schemas.prediction import LanguageInfo, PredictionResult


class EchoModel(BaseNALTRAModel):
    def train(self, train_data: Any, validation_data: Any | None = None) -> None:
        del train_data, validation_data

    def predict(self, text: str) -> PredictionResult:
        return PredictionResult(
            text=text,
            model="echo",
            language=LanguageInfo(primary="und"),
        )

    def predict_batch(self, texts: Iterable[str]) -> list[PredictionResult]:
        return [self.predict(text) for text in texts]

    def save(self, path: str | Path) -> None:
        del path

    def load(self, path: str | Path) -> None:
        del path


def test_pipeline_normalizes_text_before_prediction() -> None:
    pipeline = PredictionPipeline(model=EchoModel())

    result = pipeline.predict("  English   ve Türkçe  ")

    assert result.text == "English ve Türkçe"


def test_pipeline_preserves_batch_order() -> None:
    pipeline = PredictionPipeline(model=EchoModel())
    results = pipeline.predict_batch([" first ", "second"])

    assert [result.text for result in results] == ["first", "second"]
