"""Cross-dataset public sleep evaluation."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..datasets.public_sleep import SleepRecord
from .evaluate import evaluate_predictions, evaluate_probabilities
from .train_xgb import (
    build_classifier,
    fit_with_subject_calibration,
    make_model_bundle,
    predict_class_probabilities,
    records_to_feature_dataset,
)


def cross_dataset_evaluate(
    train_records: Sequence[SleepRecord],
    test_records: Sequence[SleepRecord],
    task: str | int = "5class",
    random_state: int = 42,
    **dataset_kwargs,
) -> dict:
    train_data = records_to_feature_dataset(train_records, task=task, **dataset_kwargs)
    test_data = records_to_feature_dataset(test_records, task=task, **dataset_kwargs)
    if train_data.feature_names != test_data.feature_names:
        raise ValueError("train and test datasets produced different feature schemas")
    model, model_kind = build_classifier(random_state=random_state)
    model, calibration = fit_with_subject_calibration(
        model,
        train_data.X,
        train_data.y,
        train_data.subject_ids,
        np.ones(train_data.y.size, dtype=bool),
        random_state,
    )
    probabilities = predict_class_probabilities(model, test_data.X, len(test_data.classes))
    y_pred = np.argmax(probabilities, axis=1)
    metrics = evaluate_predictions(test_data.y, y_pred, test_data.classes, test_data.subject_ids)
    metrics.update(evaluate_probabilities(test_data.y, probabilities))
    metrics.update(
        {
            "model_kind": model_kind,
            "train_dataset": sorted({record.dataset for record in train_records}),
            "test_dataset": sorted({record.dataset for record in test_records}),
            "n_train_epochs": int(train_data.y.size),
            "n_test_epochs": int(test_data.y.size),
            "calibration": calibration,
            "train_synthetic_demo": bool(train_data.metadata.get("synthetic_demo")),
            "test_synthetic_demo": bool(test_data.metadata.get("synthetic_demo")),
        }
    )
    bundle = make_model_bundle(model, model_kind, train_data)
    bundle["calibration"] = calibration
    return {"bundle": bundle, "metrics": metrics}


def bidirectional_cross_dataset(
    sleep_edf_records: Sequence[SleepRecord],
    isruc_records: Sequence[SleepRecord],
    task: str | int = "5class",
    **dataset_kwargs,
) -> dict:
    return {
        "sleep_edf_to_isruc": cross_dataset_evaluate(sleep_edf_records, isruc_records, task=task, **dataset_kwargs)[
            "metrics"
        ],
        "isruc_to_sleep_edf": cross_dataset_evaluate(isruc_records, sleep_edf_records, task=task, **dataset_kwargs)[
            "metrics"
        ],
    }
