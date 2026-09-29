"""Boundaries for reproducible noise and adversarial transformations."""


def perturb_text(text: str, strategy: str) -> str:
    """Create a controlled perturbation using a future named strategy."""
    del text, strategy
    raise NotImplementedError("Noise generation has not been implemented.")
