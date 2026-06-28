"""Leakage-safe traditional baseline for real public sleep EEG."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold


FIVE_CLASS_LABELS = ["W", "N1", "N2", "N3", "REM"]
THREE_CLASS_LABELS = ["Wake", "NREM", "REM"]
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (16.0, 30.0),
}


def map_labels(labels: Sequence[str], task: str) -> np.ndarray:
    labels = np.asarray(labels, dtype=object)
    if task == "5class":
        mapped = labels.astype(str)
        if not set(mapped).issubset(FIVE_CLASS_LABELS):
            raise ValueError("5-class labels must be W/N1/N2/N3/REM")
        return mapped
    if task == "3class":
        mapping = {"W": "Wake", "N1": "NREM", "N2": "NREM", "N3": "NREM", "REM": "REM"}
        try:
            return np.asarray([mapping[str(label)] for label in labels], dtype=str)
        except KeyError as exc:
            raise ValueError(f"unsupported canonical sleep label: {exc.args[0]}") from exc
    raise ValueError(f"unsupported task: {task}")


def extract_traditional_features(
    epochs: np.ndarray,
    sfreq: float,
    channel_names: Sequence[str] = ("Fpz-Cz", "Pz-Oz"),
) -> tuple[np.ndarray, list[str]]:
    """Extract requested time, Hjorth, spectral power, ratio and entropy features."""

    epochs = np.asarray(epochs, dtype=float)
    if epochs.ndim != 3 or epochs.shape[1] != len(channel_names):
        raise ValueError("epochs must have shape (epochs, channels, samples)")
    rows: list[np.ndarray] = []
    names: list[str] = []
    for channel_index, channel_name in enumerate(channel_names):
        x = np.nan_to_num(epochs[:, channel_index, :])
        dx = np.diff(x, axis=1)
        ddx = np.diff(dx, axis=1)
        activity = np.var(x, axis=1)
        mobility = np.sqrt(np.var(dx, axis=1) / np.maximum(activity, 1e-24))
        mobility_dx = np.sqrt(np.var(ddx, axis=1) / np.maximum(np.var(dx, axis=1), 1e-24))
        complexity = mobility_dx / np.maximum(mobility, 1e-24)
        time_features = [
            np.mean(x, axis=1),
            np.std(x, axis=1),
            np.ptp(x, axis=1),
            activity,
            mobility,
            complexity,
            np.sum(np.abs(dx), axis=1),
            np.sum((x[:, :-1] * x[:, 1:]) < 0, axis=1),
        ]
        prefix = _safe_name(channel_name)
        time_names = [
            "mean",
            "std",
            "ptp",
            "hjorth_activity",
            "hjorth_mobility",
            "hjorth_complexity",
            "line_length",
            "zero_crossings",
        ]
        rows.extend(time_features)
        names.extend(f"{prefix}_{name}" for name in time_names)

        nperseg = min(x.shape[1], max(64, int(round(sfreq * 4))))
        frequencies, psd = welch(x, fs=sfreq, nperseg=nperseg, axis=1)
        total_mask = (frequencies >= 0.5) & (frequencies <= 35.0)
        total_power = np.trapezoid(psd[:, total_mask], frequencies[total_mask], axis=1)
        total_power = np.maximum(total_power, 1e-24)
        band_power: dict[str, np.ndarray] = {}
        for band, (low, high) in BANDS.items():
            mask = (frequencies >= low) & (frequencies < high)
            power = np.trapezoid(psd[:, mask], frequencies[mask], axis=1)
            band_power[band] = power
            rows.extend([power, power / total_power])
            names.extend([f"{prefix}_{band}_power", f"{prefix}_{band}_relative_power"])
        rows.extend(
            [
                band_power["theta"] / np.maximum(band_power["alpha"], 1e-24),
                band_power["delta"] / np.maximum(band_power["beta"], 1e-24),
            ]
        )
        names.extend([f"{prefix}_theta_alpha_ratio", f"{prefix}_delta_beta_ratio"])
        normalized_psd = psd[:, total_mask] / np.maximum(np.sum(psd[:, total_mask], axis=1, keepdims=True), 1e-24)
        spectral_entropy = -np.sum(normalized_psd * np.log2(np.maximum(normalized_psd, 1e-24)), axis=1)
        rows.append(spectral_entropy)
        names.append(f"{prefix}_spectral_entropy")
    matrix = np.column_stack(rows)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0), names


def grouped_random_forest_baseline(
    features: np.ndarray,
    labels_5class: Sequence[str],
    subject_ids: Sequence[str],
    task: str,
    random_state: int = 42,
) -> tuple[dict, list[dict], list[dict]]:
    """Generate pooled out-of-fold predictions using subject-only GroupKFold splits."""

    features = np.asarray(features, dtype=float)
    subjects = np.asarray(subject_ids, dtype=str)
    labels = map_labels(labels_5class, task)
    unique_subjects = np.unique(subjects)
    if unique_subjects.size < 2:
        raise ValueError("at least two subjects are required for grouped evaluation")
    classes = FIVE_CLASS_LABELS if task == "5class" else THREE_CLASS_LABELS
    splitter = GroupKFold(n_splits=min(5, unique_subjects.size))
    predictions = np.empty(labels.shape, dtype=object)
    confidences = np.zeros(labels.shape, dtype=float)
    fold_by_epoch = np.zeros(labels.shape, dtype=int)
    prediction_rows: list[dict] = []
    split_rows: list[dict] = []
    train_sizes: list[int] = []
    test_sizes: list[int] = []
    for fold, (train_index, test_index) in enumerate(splitter.split(features, labels, groups=subjects), start=1):
        train_subjects = sorted(set(subjects[train_index]))
        test_subjects = sorted(set(subjects[test_index]))
        overlap = set(train_subjects) & set(test_subjects)
        if overlap:
            raise RuntimeError(f"subject leakage detected in fold {fold}: {sorted(overlap)}")
        model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=random_state + fold,
            n_jobs=-1,
            min_samples_leaf=2,
        )
        model.fit(features[train_index], labels[train_index])
        fold_predictions = model.predict(features[test_index])
        probabilities = model.predict_proba(features[test_index])
        predictions[test_index] = fold_predictions
        confidences[test_index] = probabilities.max(axis=1)
        fold_by_epoch[test_index] = fold
        train_sizes.append(len(train_index))
        test_sizes.append(len(test_index))
        split_rows.append(
            {
                "fold": fold,
                "train_subject_ids": "|".join(train_subjects),
                "test_subject_ids": "|".join(test_subjects),
                "n_train_epochs": len(train_index),
                "n_test_epochs": len(test_index),
                "subject_overlap_count": 0,
            }
        )
    for index, (subject, truth, predicted, confidence, fold) in enumerate(
        zip(subjects, labels, predictions, confidences, fold_by_epoch)
    ):
        prediction_rows.append(
            {
                "epoch_row": index,
                "subject_id": subject,
                "fold": int(fold),
                "true_label": str(truth),
                "predicted_label": str(predicted),
                "confidence": float(confidence),
            }
        )
    report = classification_report(labels, predictions, labels=classes, output_dict=True, zero_division=0)
    class_distribution = Counter(map(str, labels))
    metrics = {
        "task": task,
        "real_public_dataset": True,
        "synthetic_demo": False,
        "n_subjects": int(unique_subjects.size),
        "subject_ids": unique_subjects.tolist(),
        "n_epochs_total": int(labels.size),
        "n_epochs_train": int(round(float(np.mean(train_sizes)))),
        "n_epochs_test": int(round(float(np.mean(test_sizes)))),
        "n_epochs_train_per_fold": train_sizes,
        "n_epochs_test_per_fold": test_sizes,
        "class_mapping": (
            {"W": "Wake", "N1": "NREM", "N2": "NREM", "N3": "NREM", "REM": "REM"}
            if task == "3class"
            else {label: label for label in FIVE_CLASS_LABELS}
        ),
        "class_distribution": {label: int(class_distribution.get(label, 0)) for label in classes},
        "split_method": f"GroupKFold(n_splits={len(split_rows)}, group=subject_id)",
        "subject_overlap_in_any_fold": False,
        "classes": classes,
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=classes, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, labels=classes, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(labels, predictions, labels=classes)),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in classes
        },
        "confusion_matrix": confusion_matrix(labels, predictions, labels=classes).astype(int).tolist(),
        "folds": split_rows,
    }
    validate_metrics_schema(metrics)
    return metrics, prediction_rows, split_rows


def validate_metrics_schema(metrics: dict) -> None:
    required = {
        "n_subjects",
        "subject_ids",
        "n_epochs_total",
        "n_epochs_train",
        "n_epochs_test",
        "class_mapping",
        "class_distribution",
        "split_method",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "cohen_kappa",
        "per_class",
        "confusion_matrix",
        "real_public_dataset",
        "synthetic_demo",
    }
    missing = sorted(required - set(metrics))
    if missing:
        raise ValueError(f"metrics schema is missing fields: {missing}")
    if metrics["synthetic_demo"] or not metrics["real_public_dataset"]:
        raise ValueError("real baseline metrics must be explicitly non-synthetic")
    if metrics.get("subject_overlap_in_any_fold"):
        raise ValueError("subject overlap is not allowed")


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name)).strip("_").lower() or "channel"


__all__ = [
    "FIVE_CLASS_LABELS",
    "THREE_CLASS_LABELS",
    "extract_traditional_features",
    "grouped_random_forest_baseline",
    "map_labels",
    "validate_metrics_schema",
]
