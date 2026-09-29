import pytest

from naltra.schemas.prediction import (
    Explanation,
    LabelScore,
    LanguageInfo,
    OODResult,
    PredictionResult,
)


def test_prediction_schema_constructs_expected_payload() -> None:
    result = PredictionResult(
        text="Şirket yeni bir AI chip geliştirdi.",
        model="transformer",
        language=LanguageInfo(primary="tr", is_code_switched=True),
        labels=[
            LabelScore(label="Technology", score=0.97),
            LabelScore(label="Hardware", score=0.95),
            LabelScore(label="Semiconductors", score=0.92),
        ],
        hierarchy_paths=[["Technology", "Hardware", "Semiconductors"]],
        ood=OODResult(is_ood=False, score=0.12),
        latency_ms=42.5,
        explanation=Explanation(important_tokens=["AI", "chip"]),
    )

    payload = result.to_dict()
    assert payload["model"] == "transformer"
    assert payload["language"] == {"primary": "tr", "is_code_switched": True}
    assert payload["labels"][0] == {"label": "Technology", "score": 0.97}
    assert payload["hierarchy_paths"][0][-1] == "Semiconductors"
    assert payload["explanation"]["important_tokens"] == ["AI", "chip"]


def test_prediction_schema_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError, match="between"):
        LabelScore(label="Technology", score=1.1)
