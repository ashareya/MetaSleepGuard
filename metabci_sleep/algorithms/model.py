"""Trainable sleep staging estimator using the project's tested classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

from MetaSleepGuard.models.train_xgb import build_classifier, predict_class_probabilities
from MetaSleepGuard.preprocessing.label_mapping import normalize_task, task_classes

from .features import SleepFeatureExtractor


class SleepStagingEstimator:
    """Fit and predict sleep stages from raw epochs or feature matrices."""

    def __init__(
        self,
        task: str = "5class",
        sfreq: float = 250.0,
        channel_names: Sequence[str] | None = None,
        context_history: int = 2,
        random_state: int = 42,
        estimator=None,
    ) -> None:
        self.task = normalize_task(task)
        self.classes = task_classes(self.task)
        self.extractor = SleepFeatureExtractor(
            sfreq=sfreq,
            channel_names=channel_names,
            context_history=context_history,
        )
        self.random_state = int(random_state)
        if estimator is None:
            self.model, self.model_kind = build_classifier(self.random_state)
        else:
            self.model, self.model_kind = estimator, type(estimator).__name__
        self._fitted = False

    def _features(self, X: np.ndarray, subject_ids=None, fit: bool = False) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            return X
        if X.ndim != 3:
            raise ValueError("X must be feature matrix or (epochs, channels, samples)")
        return self.extractor.fit_transform(X, subject_ids=subject_ids) if fit else self.extractor.transform(
            X, subject_ids=subject_ids
        )

    def fit(self, X: np.ndarray, y: Sequence[int], subject_ids: Sequence[str] | None = None):
        features = self._features(X, subject_ids=subject_ids, fit=True)
        labels = np.asarray(y, dtype=int)
        if len(labels) != len(features):
            raise ValueError("X and y row counts must match")
        self.model.fit(features, labels)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray, subject_ids: Sequence[str] | None = None) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("estimator is not fitted")
        features = self._features(X, subject_ids=subject_ids)
        return predict_class_probabilities(self.model, features, len(self.classes))

    def predict(self, X: np.ndarray, subject_ids: Sequence[str] | None = None) -> np.ndarray:
        return np.argmax(self.predict_proba(X, subject_ids=subject_ids), axis=1)

    def predict_labels(self, X: np.ndarray, subject_ids: Sequence[str] | None = None) -> list[str]:
        return [self.classes[index] for index in self.predict(X, subject_ids=subject_ids)]

    def predict_proba_stream(
        self,
        X: np.ndarray,
        subject_ids: Sequence[str] | None = None,
    ) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("estimator is not fitted")
        array = np.asarray(X, dtype=float)
        if array.ndim == 2:
            features = array
        elif array.ndim == 3:
            features = self.extractor.transform_stream(array, subject_ids=subject_ids)
        else:
            raise ValueError("X must be feature matrix or (epochs, channels, samples)")
        return predict_class_probabilities(self.model, features, len(self.classes))

    def predict_stream(self, X: np.ndarray, subject_ids: Sequence[str] | None = None) -> np.ndarray:
        return np.argmax(self.predict_proba_stream(X, subject_ids=subject_ids), axis=1)

    def predict_labels_stream(self, X: np.ndarray, subject_ids: Sequence[str] | None = None) -> list[str]:
        return [self.classes[index] for index in self.predict_stream(X, subject_ids=subject_ids)]

    def reset_stream(self, subject_id: str | None = None) -> None:
        self.extractor.reset_stream(subject_id)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SleepStagingEstimator":
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError("saved object is not a SleepStagingEstimator")
        return loaded
