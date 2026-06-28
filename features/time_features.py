"""Time-domain EEG features."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def time_domain_features(epoch: np.ndarray, channel_names: Sequence[str]) -> dict[str, float]:
    epoch = np.asarray(epoch, dtype=float)
    features: dict[str, float] = {}
    for i, name in enumerate(channel_names):
        x = np.nan_to_num(epoch[i])
        centered = x - np.mean(x)
        std = float(np.std(x))
        if std > 0:
            skew = float(np.mean((centered / std) ** 3))
            kurtosis = float(np.mean((centered / std) ** 4) - 3.0)
        else:
            skew = 0.0
            kurtosis = 0.0
        prefix = _safe_name(name)
        features[f"{prefix}_mean"] = float(np.mean(x))
        features[f"{prefix}_var"] = float(np.var(x))
        features[f"{prefix}_std"] = std
        features[f"{prefix}_skew"] = skew
        features[f"{prefix}_kurtosis"] = kurtosis
        features[f"{prefix}_ptp"] = float(np.ptp(x))
        features[f"{prefix}_rms"] = float(np.sqrt(np.mean(x**2)))
    return features


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name)).strip("_").lower() or "ch"

