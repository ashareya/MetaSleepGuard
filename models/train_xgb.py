"""XGBoost baseline training with subject-level splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import logging
from typing import Sequence

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ..datasets.public_sleep import SleepRecord
from ..features import extract_features_from_epochs
from ..features.causal_context import append_causal_context, causal_feature_names
from ..preprocessing.channel_select import choose_two_channels, select_channels
from ..preprocessing.epoching import align_epoch_labels, epoch_signal, subject_level_split
from ..preprocessing.filters import preprocess_signal
from ..preprocessing.label_mapping import encode_labels, normalize_task
from ..rejection.calibration import calibrate_classifier
from .evaluate import evaluate_predictions, evaluate_probabilities

LOGGER = logging.getLogger(__name__)


@dataclass
class FeatureDataset:
    X: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    classes: list[str]
    feature_names: list[str]
    base_feature_names: list[str]
    metadata: dict


def records_to_feature_dataset(
    records: Sequence[SleepRecord],
    task: str | int = "5class",
    channels: Sequence[str] | None = None,
    target_sfreq: float = 250.0,
    bandpass: tuple[float, float] = (0.3, 35.0),
    context_history: int = 2,
    n_channels: int = 2,
    apply_preprocessing: bool = True,
) -> FeatureDataset:
    """Convert subject records to causal-context feature matrix."""

    rows: list[np.ndarray] = []
    labels_all: list[int] = []
    subjects_all: list[str] = []
    base_names: list[str] | None = None
    classes: list[str] | None = None
    kept_epochs = 0
    dropped_epochs = 0
    for record in records:
        if channels:
            signals, names = select_channels(record.signals, record.channel_names, channels)
        else:
            signals, names = choose_two_channels(record.signals, record.channel_names)
        if n_channels not in {1, 2}:
            raise ValueError("n_channels must be 1 or 2")
        if signals.shape[0] < n_channels:
            raise ValueError(f"record {record.subject_id} has fewer than {n_channels} selected EEG channels")
        signals = signals[:n_channels]
        names = [f"EEG{i + 1}" for i in range(n_channels)]
        sfreq = record.sfreq
        if apply_preprocessing:
            signals, sfreq = preprocess_signal(signals, sfreq, target_sfreq=target_sfreq, bandpass=bandpass)
        epochs = epoch_signal(signals, sfreq, epoch_sec=record.epoch_sec)
        labels = align_epoch_labels(record.labels, epochs.shape[0])
        encoded, task_classes, keep = encode_labels(labels, task)
        epoch_features, feature_names = extract_features_from_epochs(epochs, sfreq, names)
        if base_names is None:
            base_names = feature_names
        elif feature_names != base_names:
            raise ValueError("feature schema changed between records")
        classes = task_classes
        keep_mask = np.asarray(keep, dtype=bool)
        dropped_epochs += int(np.sum(~keep_mask))
        kept_epochs += int(np.sum(keep_mask))
        if np.any(keep_mask):
            epoch_context = append_causal_context(
                epoch_features,
                [record.subject_id] * epoch_features.shape[0],
                history=context_history,
            )
            rows.append(epoch_context[keep_mask])
            labels_all.extend(np.asarray(encoded)[keep_mask].tolist())
            subjects_all.extend([record.subject_id] * int(np.sum(keep_mask)))
    if not rows:
        raise ValueError("no labeled epochs were available after label normalization")
    X = np.vstack(rows)
    final_feature_names = causal_feature_names(base_names or [], history=context_history)
    return FeatureDataset(
        X=np.nan_to_num(X),
        y=np.asarray(labels_all, dtype=int),
        subject_ids=np.asarray(subjects_all),
        classes=classes or [],
        feature_names=final_feature_names,
        base_feature_names=base_names or [],
        metadata={
            "task": normalize_task(task),
            "datasets": sorted({record.dataset for record in records}),
            "synthetic_demo": any(bool(record.metadata.get("synthetic")) for record in records),
            "n_subjects": len({record.subject_id for record in records}),
            "target_sfreq": target_sfreq,
            "context_history": context_history,
            "n_channels": n_channels,
            "kept_epochs": kept_epochs,
            "dropped_epochs": dropped_epochs,
            "preprocessing": {"bandpass": list(bandpass), "notch_hz": 50.0},
        },
    )


def build_classifier(random_state: int = 42):
    """Create an XGBoost classifier, falling back to sklearn when unavailable."""

    try:
        from xgboost import XGBClassifier  # type: ignore

        return XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=random_state,
        ), "xgboost"
    except Exception as exc:
        LOGGER.warning("xgboost is unavailable or incompatible (%s); using RandomForestClassifier fallback", exc)
        return RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ), "sklearn_random_forest_fallback"


def train_subject_split(
    records: Sequence[SleepRecord],
    task: str | int = "5class",
    output_path: str | Path | None = None,
    test_fraction: float = 0.25,
    random_state: int = 42,
    **dataset_kwargs,
) -> dict:
    """Train and evaluate a baseline using a subject-level train/test split."""

    dataset = records_to_feature_dataset(records, task=task, **dataset_kwargs)
    train_mask, test_mask = subject_level_split(dataset.subject_ids, test_fraction=test_fraction, seed=random_state)
    model, model_kind = build_classifier(random_state=random_state)
    model, calibration = fit_with_subject_calibration(
        model,
        dataset.X,
        dataset.y,
        dataset.subject_ids,
        train_mask,
        random_state,
    )
    probabilities = predict_class_probabilities(model, dataset.X[test_mask], len(dataset.classes))
    y_pred = np.argmax(probabilities, axis=1)
    metrics = evaluate_predictions(dataset.y[test_mask], y_pred, dataset.classes, dataset.subject_ids[test_mask])
    metrics.update(evaluate_probabilities(dataset.y[test_mask], probabilities))
    bundle = make_model_bundle(model, model_kind, dataset)
    bundle["calibration"] = calibration
    if output_path:
        save_model_bundle(bundle, output_path)
        metrics["model_path"] = str(output_path)
    metrics["model_kind"] = model_kind
    metrics["n_train_epochs"] = int(np.sum(train_mask))
    metrics["n_test_epochs"] = int(np.sum(test_mask))
    metrics["calibration"] = calibration
    metrics["metadata"] = dataset.metadata
    return {"bundle": bundle, "metrics": metrics, "dataset": dataset}


def make_model_bundle(model, model_kind: str, dataset: FeatureDataset) -> dict:
    return {
        "model": model,
        "model_kind": model_kind,
        "classes": dataset.classes,
        "feature_names": dataset.feature_names,
        "base_feature_names": dataset.base_feature_names,
        "metadata": dataset.metadata,
    }


def save_model_bundle(bundle: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    meta_path = path.with_suffix(path.suffix + ".json")
    meta_path.write_text(
        json.dumps({k: v for k, v in bundle.items() if k != "model"}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def load_model_bundle(path: str | Path) -> dict:
    return joblib.load(path)


def predict_class_probabilities(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    """Return probabilities aligned to integer labels ``0..n_classes-1``."""

    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    model_classes = np.asarray(model.classes_, dtype=int)
    if probabilities.shape[1] == n_classes and np.array_equal(model_classes, np.arange(n_classes)):
        return probabilities
    aligned = np.zeros((probabilities.shape[0], n_classes), dtype=float)
    valid = (model_classes >= 0) & (model_classes < n_classes)
    aligned[:, model_classes[valid]] = probabilities[:, valid]
    row_sum = aligned.sum(axis=1, keepdims=True)
    return np.divide(aligned, row_sum, out=np.full_like(aligned, 1.0 / n_classes), where=row_sum > 0)


def fit_with_subject_calibration(model, X, y, subject_ids, train_mask, random_state: int):
    train_indices = np.flatnonzero(train_mask)
    train_subjects = np.asarray(subject_ids)[train_indices]
    unique_subjects = np.unique(train_subjects)
    if unique_subjects.size < 3:
        model.fit(X[train_indices], y[train_indices])
        return model, {"applied": False, "reason": "fewer_than_three_training_subjects"}

    fit_local, calibration_local = subject_level_split(train_subjects, test_fraction=0.25, seed=random_state + 1)
    fit_indices = train_indices[fit_local]
    calibration_indices = train_indices[calibration_local]
    if np.unique(y[fit_indices]).size < np.unique(y[train_indices]).size:
        model.fit(X[train_indices], y[train_indices])
        return model, {"applied": False, "reason": "fit_subjects_do_not_cover_all_classes"}
    if np.unique(y[calibration_indices]).size < np.unique(y[train_indices]).size:
        model.fit(X[train_indices], y[train_indices])
        return model, {"applied": False, "reason": "calibration_subjects_do_not_cover_all_classes"}

    model.fit(X[fit_indices], y[fit_indices])
    try:
        calibrated = calibrate_classifier(model, X[calibration_indices], y[calibration_indices])
    except Exception as exc:
        LOGGER.warning("probability calibration failed; using uncalibrated model: %s", exc)
        model.fit(X[train_indices], y[train_indices])
        return model, {"applied": False, "reason": f"calibration_failed:{type(exc).__name__}"}
    return calibrated, {
        "applied": True,
        "method": "sigmoid",
        "split": "held_out_subjects",
        "fit_subjects": sorted(set(np.asarray(subject_ids)[fit_indices].tolist())),
        "calibration_subjects": sorted(set(np.asarray(subject_ids)[calibration_indices].tolist())),
        "n_fit_epochs": int(fit_indices.size),
        "n_calibration_epochs": int(calibration_indices.size),
    }
