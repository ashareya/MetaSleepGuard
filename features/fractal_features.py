"""Fractal features."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def petrosian_fd_features(epoch: np.ndarray, channel_names: Sequence[str]) -> dict[str, float]:
    epoch = np.asarray(epoch, dtype=float)
    features: dict[str, float] = {}
    for i, name in enumerate(channel_names):
        features[f"{_safe_name(name)}_petrosian_fd"] = float(petrosian_fd(epoch[i]))
    return features


def petrosian_fd(x: np.ndarray) -> float:
    x = np.nan_to_num(np.asarray(x, dtype=float))
    n = x.size
    if n < 2:
        return 0.0
    diff = np.diff(x)
    sign_changes = np.sum(diff[1:] * diff[:-1] < 0)
    if sign_changes <= 0:
        return 1.0
    return float(np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * sign_changes))))


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name)).strip("_").lower() or "ch"

