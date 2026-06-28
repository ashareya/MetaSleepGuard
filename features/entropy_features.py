"""Entropy features."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def shannon_entropy_features(epoch: np.ndarray, channel_names: Sequence[str], bins: int = 64) -> dict[str, float]:
    epoch = np.asarray(epoch, dtype=float)
    features: dict[str, float] = {}
    for i, name in enumerate(channel_names):
        x = np.nan_to_num(epoch[i])
        hist, _ = np.histogram(x, bins=bins, density=False)
        prob = hist.astype(float)
        prob = prob[prob > 0] / max(1.0, float(np.sum(prob)))
        entropy = -float(np.sum(prob * np.log2(prob))) if prob.size else 0.0
        features[f"{_safe_name(name)}_shannon_entropy"] = entropy
    return features


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name)).strip("_").lower() or "ch"

