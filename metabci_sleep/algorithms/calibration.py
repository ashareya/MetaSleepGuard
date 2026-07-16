"""Probability calibration and coverage-risk evaluation wrappers."""

from __future__ import annotations

import numpy as np

from MetaSleepGuard.models.evaluate import evaluate_probabilities
from MetaSleepGuard.rejection.calibration import calibrate_classifier
from MetaSleepGuard.rejection.coverage_risk import coverage_risk_curve, macro_f1_before_after_rejection


class ProbabilityCalibrator:
    def __init__(self, method: str = "sigmoid") -> None:
        self.method = method
        self.model = None

    def fit(self, base_estimator, X: np.ndarray, y: np.ndarray):
        self.model = calibrate_classifier(base_estimator, np.asarray(X), np.asarray(y), method=self.method)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("calibrator is not fitted")
        return np.asarray(self.model.predict_proba(X), dtype=float)

    @staticmethod
    def evaluate(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
        return evaluate_probabilities(np.asarray(y_true), np.asarray(probabilities))


class CoverageRiskEvaluator:
    def curve(self, y_true, probabilities, thresholds=None) -> list[dict]:
        return coverage_risk_curve(y_true, probabilities, thresholds=thresholds)

    def rejection_summary(self, y_true, y_pred, accepted, n_classes: int) -> dict:
        return macro_f1_before_after_rejection(y_true, y_pred, accepted, n_classes)
