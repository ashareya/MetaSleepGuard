"""Calibration metrics and optional calibrated classifier wrapper."""

from __future__ import annotations

import numpy as np


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    confidence = np.max(probs, axis=1)
    prediction = np.argmax(probs, axis=1)
    correctness = prediction == y_true
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidence > low) & (confidence <= high)
        if np.any(mask):
            acc = np.mean(correctness[mask])
            conf = np.mean(confidence[mask])
            ece += np.mean(mask) * abs(acc - conf)
    return float(ece)


def multiclass_brier_score(y_true: np.ndarray, probabilities: np.ndarray, n_classes: int | None = None) -> float:
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    n_classes = n_classes or probs.shape[1]
    one_hot = np.zeros((y_true.size, n_classes), dtype=float)
    one_hot[np.arange(y_true.size), y_true] = 1.0
    return float(np.mean(np.sum((probs[:, :n_classes] - one_hot) ** 2, axis=1)))


def calibrate_classifier(base_estimator, X_cal: np.ndarray, y_cal: np.ndarray, method: str = "sigmoid"):
    """Return a sklearn CalibratedClassifierCV fitted on calibration data."""

    from sklearn.calibration import CalibratedClassifierCV

    try:
        from sklearn.frozen import FrozenEstimator

        indices = np.arange(len(y_cal), dtype=int)
        calibrated = CalibratedClassifierCV(
            FrozenEstimator(base_estimator),
            method=method,
            cv=[(indices, indices)],
            ensemble=False,
        )
    except ImportError:  # pragma: no cover - sklearn < 1.6
        calibrated = CalibratedClassifierCV(base_estimator, method=method, cv="prefit")
    calibrated.fit(X_cal, y_cal)
    return calibrated
