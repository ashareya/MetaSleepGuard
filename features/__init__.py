"""Feature extraction for 30-second two-channel sleep EEG epochs."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .causal_context import append_causal_context
from .entropy_features import shannon_entropy_features
from .fractal_features import petrosian_fd_features
from .frequency_features import bandpower_features
from .hjorth_features import hjorth_features
from .time_features import time_domain_features


def extract_epoch_features(epoch: np.ndarray, sfreq: float, channel_names: Sequence[str] | None = None) -> dict[str, float]:
    """Extract all baseline features from one epoch.

    Parameters
    ----------
    epoch:
        Array with shape ``(channels, samples)``.
    sfreq:
        Sampling frequency in Hz.
    channel_names:
        Optional channel labels used in feature names.
    """

    epoch = np.asarray(epoch, dtype=float)
    if epoch.ndim != 2:
        raise ValueError("epoch must have shape (channels, samples)")
    names = list(channel_names or [f"ch{i + 1}" for i in range(epoch.shape[0])])
    features: dict[str, float] = {}
    features.update(time_domain_features(epoch, names))
    features.update(bandpower_features(epoch, sfreq, names))
    features.update(hjorth_features(epoch, names))
    features.update(shannon_entropy_features(epoch, names))
    features.update(petrosian_fd_features(epoch, names))
    if epoch.shape[0] >= 2:
        if np.std(epoch[0]) > 0 and np.std(epoch[1]) > 0:
            corr = float(np.corrcoef(epoch[0], epoch[1])[0, 1])
        else:
            corr = 0.0
        features["ch1_ch2_corr"] = float(np.nan_to_num(corr))
    return features


def extract_features_from_epochs(
    epochs: np.ndarray,
    sfreq: float,
    channel_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return a numeric feature matrix for many epochs."""

    rows = [extract_epoch_features(epoch, sfreq, channel_names) for epoch in np.asarray(epochs)]
    if not rows:
        return np.empty((0, 0), dtype=float), []
    names = sorted(rows[0].keys())
    matrix = np.array([[row.get(name, 0.0) for name in names] for row in rows], dtype=float)
    return np.nan_to_num(matrix), names


__all__ = [
    "append_causal_context",
    "bandpower_features",
    "extract_epoch_features",
    "extract_features_from_epochs",
    "hjorth_features",
    "petrosian_fd_features",
    "shannon_entropy_features",
    "time_domain_features",
]
