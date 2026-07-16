"""MetaBCI-facing multi-dimensional sleep feature extraction."""

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np

from MetaSleepGuard.features import extract_features_from_epochs
from MetaSleepGuard.features.causal_context import append_causal_context, causal_feature_names


class SleepFeatureExtractor:
    """Extract tested time, frequency, Hjorth, entropy, and fractal features."""

    def __init__(
        self,
        sfreq: float,
        channel_names: Sequence[str] | None = None,
        context_history: int = 2,
        input_unit: str = "microvolts",
    ) -> None:
        if sfreq <= 0:
            raise ValueError("sfreq must be positive")
        if context_history < 0:
            raise ValueError("context_history must be non-negative")
        if input_unit not in {"microvolts", "volts"}:
            raise ValueError("input_unit must be 'microvolts' or 'volts'")
        self.sfreq = float(sfreq)
        self.channel_names = None if channel_names is None else list(channel_names)
        self.context_history = int(context_history)
        self.input_unit = input_unit
        self.base_feature_names_: list[str] = []
        self.feature_names_: list[str] = []
        self._stream_history: dict[str, deque[np.ndarray]] = {}

    def transform(
        self,
        epochs: np.ndarray,
        subject_ids: Sequence[str] | None = None,
    ) -> np.ndarray:
        epochs = np.asarray(epochs, dtype=float)
        if epochs.ndim != 3:
            raise ValueError("epochs must have shape (epochs, channels, samples)")
        volts = epochs * 1e-6 if self.input_unit == "microvolts" else epochs
        matrix, names = extract_features_from_epochs(volts, self.sfreq, self.channel_names)
        self.base_feature_names_ = names
        if self.context_history:
            subjects = list(subject_ids or ["subject"] * len(matrix))
            matrix = append_causal_context(matrix, subjects, history=self.context_history)
            self.feature_names_ = causal_feature_names(names, history=self.context_history)
        else:
            self.feature_names_ = names
        return matrix

    def transform_stream(
        self,
        epochs: np.ndarray,
        subject_ids: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Extract causal features while preserving history across calls."""

        epochs = np.asarray(epochs, dtype=float)
        if epochs.ndim != 3:
            raise ValueError("epochs must have shape (epochs, channels, samples)")
        subjects = list(subject_ids or ["subject"] * len(epochs))
        if len(subjects) != len(epochs):
            raise ValueError("subject_ids length must match epoch rows")
        volts = epochs * 1e-6 if self.input_unit == "microvolts" else epochs
        base, names = extract_features_from_epochs(volts, self.sfreq, self.channel_names)
        self.base_feature_names_ = names
        self.feature_names_ = causal_feature_names(names, history=self.context_history)
        rows = []
        for vector, subject in zip(base, subjects):
            history = self._stream_history.setdefault(
                str(subject), deque(maxlen=self.context_history)
            )
            missing = self.context_history - len(history)
            blocks = [np.zeros_like(vector) for _ in range(missing)]
            blocks.extend(history)
            blocks.append(vector)
            rows.append(np.concatenate(blocks))
            history.append(vector.copy())
        return np.vstack(rows)

    def reset_stream(self, subject_id: str | None = None) -> None:
        if subject_id is None:
            self._stream_history.clear()
        else:
            self._stream_history.pop(str(subject_id), None)

    def fit(self, epochs: np.ndarray, y=None, subject_ids: Sequence[str] | None = None):
        del y
        self.transform(epochs, subject_ids=subject_ids)
        return self

    def fit_transform(self, epochs: np.ndarray, y=None, subject_ids: Sequence[str] | None = None):
        del y
        return self.transform(epochs, subject_ids=subject_ids)

    def get_feature_names_out(self) -> list[str]:
        if not self.feature_names_:
            raise RuntimeError("transform must be called before requesting feature names")
        return self.feature_names_.copy()
