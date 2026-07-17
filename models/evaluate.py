"""Model evaluation metrics for public sleep staging."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score

from ..rejection.calibration import expected_calibration_error, multiclass_brier_score
from ..rejection.coverage_risk import coverage_risk_curve, macro_f1_before_after_rejection


def evaluate_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    classes: Sequence[str],
    subject_ids: Sequence[str] | None = None,
) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(classes)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(classes),
        output_dict=True,
        zero_division=0,
    )
    kappa = float(cohen_kappa_score(y_true, y_pred, labels=labels))
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "cohen_kappa": float(np.nan_to_num(kappa)),
        "per_class": {name: report[name] for name in classes if name in report},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if subject_ids is not None:
        metrics["per_subject"] = evaluate_by_subject(y_true, y_pred, classes, subject_ids)
    return metrics


def evaluate_by_subject(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    classes: Sequence[str],
    subject_ids: Sequence[str],
) -> dict[str, dict]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    subjects = np.asarray(subject_ids)
    output: dict[str, dict] = {}
    for subject in np.unique(subjects):
        mask = subjects == subject
        output[str(subject)] = {
            "n_epochs": int(np.sum(mask)),
            "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
            "macro_f1": float(
                f1_score(y_true[mask], y_pred[mask], labels=list(range(len(classes))), average="macro", zero_division=0)
            ),
        }
    return output


def evaluate_probabilities(
    y_true: Sequence[int],
    probabilities: np.ndarray,
    confidence_threshold: float = 0.55,
) -> dict:
    """Evaluate calibration and confidence-based active rejection."""

    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    y_pred = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    accepted = confidence >= float(confidence_threshold)
    before_after = macro_f1_before_after_rejection(
        y_true,
        y_pred,
        accepted,
        n_classes=probabilities.shape[1],
    )
    return {
        "ece": expected_calibration_error(y_true, probabilities),
        "brier_score": multiclass_brier_score(y_true, probabilities),
        "confidence_threshold": float(confidence_threshold),
        "rejection": before_after,
        "coverage_risk_curve": coverage_risk_curve(y_true, probabilities),
    }
