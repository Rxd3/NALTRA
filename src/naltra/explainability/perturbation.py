"""Placeholder for model-agnostic perturbation explanations."""

from naltra.schemas.prediction import Explanation


def explain_by_perturbation(text: str) -> Explanation:
    del text
    raise NotImplementedError("Perturbation explainability has not been implemented.")
