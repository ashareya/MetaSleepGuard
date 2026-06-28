"""Coverage-risk analysis for active rejection."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import f1_score


def coverage_risk_curve(
    y_true: Sequence[int],
    probabilities: np.ndarray,
    thresholds: Sequence[float] | None = None,
    average: str = "macro",
) -> list[dict]:
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    thresholds = list(thresholds if thresholds is not None else np.linspace(0.0, 1.0, 21))
    confidence = np.max(probs, axis=1)
    prediction = np.argmax(probs, axis=1)
    rows: list[dict] = []
    labels = list(range(probs.shape[1]))
    for threshold in thresholds:
        accepted = confidence >= threshold
        coverage = float(np.mean(accepted)) if accepted.size else 0.0
        if np.any(accepted):
            risk = float(np.mean(prediction[accepted] != y_true[accepted]))
            macro_f1 = float(f1_score(y_true[accepted], prediction[accepted], labels=labels, average=average, zero_division=0))
        else:
            risk = 0.0
            macro_f1 = 0.0
        rows.append({"threshold": float(threshold), "coverage": coverage, "risk": risk, "macro_f1": macro_f1})
    return rows


def macro_f1_before_after_rejection(y_true: Sequence[int], y_pred: Sequence[int], accepted: Sequence[bool], n_classes: int) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    accepted = np.asarray(accepted, dtype=bool)
    labels = list(range(n_classes))
    return {
        "before_macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "after_macro_f1": float(
            f1_score(y_true[accepted], y_pred[accepted], labels=labels, average="macro", zero_division=0)
        )
        if np.any(accepted)
        else 0.0,
        "coverage": float(np.mean(accepted)) if accepted.size else 0.0,
    }

