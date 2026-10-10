"""TF-IDF + one binary multinomial Naive Bayes per label."""

import math
from collections.abc import Sequence

import numpy as np
from scipy.sparse import spmatrix
from scipy.special import expit
from sklearn.naive_bayes import MultinomialNB

from naltra.models.classical import ClassicalModel


class NaiveBayesModel(ClassicalModel):
    """Bag-of-words baseline. With the 473 rare CORDIS labels its probabilities sit near the
    label prior (underconfident), so alpha and the threshold must be tuned on validation."""

    model_name = "naive_bayes"

    def _validate_classifier(self) -> None:
        alpha = self.config["classifier"]["alpha"]
        if isinstance(alpha, bool) or not isinstance(alpha, int | float):
            raise ValueError("classifier.alpha must be a number.")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("classifier.alpha must be finite and positive.")

    def _classifier(self, groups: Sequence[str]) -> MultinomialNB:
        return MultinomialNB(alpha=self.config["classifier"]["alpha"])

    def _compact(self, estimators: Sequence[MultinomialNB]) -> dict[str, np.ndarray]:
        # P(yes | x) = sigmoid(joint log-likelihood of yes minus that of no).
        return {
            "coef": np.ascontiguousarray(
                np.stack([e.feature_log_prob_[1] - e.feature_log_prob_[0] for e in estimators]).T,
                dtype=np.float32,
            ),
            "intercept": np.array(
                [e.class_log_prior_[1] - e.class_log_prior_[0] for e in estimators]
            ),
        }

    def _score_matrix(self, features: spmatrix, texts: Sequence[str]) -> np.ndarray:
        return expit(features @ self.weights["coef"] + self.weights["intercept"])
