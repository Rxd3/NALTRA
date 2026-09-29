"""Classical-model feature importance boundary."""

from collections.abc import Sequence

from naltra.schemas.prediction import Explanation


def explain_classical(tokens: Sequence[str], weights: Sequence[float]) -> Explanation:
    """Return positively weighted tokens ordered by absolute importance."""
    if len(tokens) != len(weights):
        raise ValueError("tokens and weights must have equal length")
    ranked = sorted(zip(tokens, weights, strict=False), key=lambda item: abs(item[1]), reverse=True)
    return Explanation(important_tokens=[token for token, weight in ranked if weight > 0])
