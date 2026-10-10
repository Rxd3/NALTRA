"""TF-IDF + one Platt-calibrated linear SVM per label."""

import math
from collections.abc import Sequence

import numpy as np
from scipy.sparse import spmatrix
from scipy.special import expit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

from naltra.models.classical import ClassicalModel

Splits = list[tuple[np.ndarray, np.ndarray]]


def calibration_splits(groups: Sequence[str], folds: int) -> Splits:
    """Group-disjoint (fit, calibrate) folds: a project is never fitted in one language while
    its translation sits in the calibration holdout."""
    return list(GroupKFold(n_splits=folds).split(groups, groups=groups))


class SVMModel(ClassicalModel):
    """Linear SVM whose margins become probabilities through cross-validated Platt scaling."""

    model_name = "svm"
    calibration = "platt_cv"

    def _validate_classifier(self) -> None:
        classifier = self.config["classifier"]
        if classifier["kernel"] != "linear":
            raise ValueError("Only the linear SVM kernel scales to the CORDIS label space.")
        c = classifier["c"]
        if isinstance(c, bool) or not isinstance(c, int | float):
            raise ValueError("classifier.c must be a number.")
        if not math.isfinite(c) or c <= 0:
            raise ValueError("classifier.c must be finite and positive.")
        folds = classifier.get("calibration_cv", 3)
        if type(folds) is not int or folds < 2:
            raise ValueError("classifier.calibration_cv must be an integer of at least 2.")

    def _calibration_cv(self, groups: Sequence[str]) -> int | Splits:
        folds = self.config["classifier"].get("calibration_cv", 3)
        # Without siblings nothing can leak, so keep scikit-learn's per-label stratified folds.
        # Memory note: clone() copies the shared splits per label (~1 MB/label at 44k records);
        # switch to PredefinedSplit fold ids if training memory becomes the limit.
        return folds if len(set(groups)) == len(groups) else calibration_splits(groups, folds)

    def _classifier(self, groups: Sequence[str]) -> CalibratedClassifierCV:
        classifier = self.config["classifier"]
        return CalibratedClassifierCV(
            LinearSVC(C=classifier["c"], random_state=self.config["seed"]),
            method="sigmoid",
            cv=self._calibration_cv(groups),
        )

    def _check_trainable(self, targets: np.ndarray, groups: Sequence[str]) -> None:
        folds = self.config["classifier"].get("calibration_cv", 3)
        cv = self._calibration_cv(groups)
        # Stratified folds cover both classes once each has `folds` examples; fixed grouped
        # folds must hold both classes in every fit and calibration part.
        parts, needed = (
            ([np.arange(len(targets))], folds)
            if isinstance(cv, int)
            else ([part for split in cv for part in split], 1)
        )
        for part in parts:
            positives = targets[part].sum(axis=0)
            if min(positives.min(), len(part) - positives.max()) < needed:
                raise ValueError(
                    f"Platt calibration over {folds} folds needs positive and negative examples "
                    "of every label in each fold's fit and calibration part."
                )

    def _compact(self, estimators: Sequence[CalibratedClassifierCV]) -> dict[str, np.ndarray]:
        # One (SVM, sigmoid) pair per calibration fold; scikit-learn averages the folds.
        folds = [list(e.calibrated_classifiers_) for e in estimators]

        def stack(read):
            return np.array([[read(fold) for fold in label] for label in folds]).swapaxes(0, 1)

        return {
            "coef": np.ascontiguousarray(
                stack(lambda f: f.estimator.coef_[0]).swapaxes(1, 2), dtype=np.float32
            ),
            "intercept": stack(lambda f: f.estimator.intercept_[0]),
            "platt_a": stack(lambda f: f.calibrators[0].a_),
            "platt_b": stack(lambda f: f.calibrators[0].b_),
        }

    def _score_matrix(self, features: spmatrix, texts: Sequence[str]) -> np.ndarray:
        w = self.weights
        per_fold = [
            expit(-(a * (features @ coef + intercept) + b))
            for coef, intercept, a, b in zip(
                w["coef"], w["intercept"], w["platt_a"], w["platt_b"], strict=True
            )
        ]
        return np.mean(per_fold, axis=0)
