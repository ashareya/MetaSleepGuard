"""Hjorth activity, mobility, and complexity."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def hjorth_features(epoch: np.ndarray, channel_names: Sequence[str]) -> dict[str, float]:
    epoch = np.asarray(epoch, dtype=float)
    features: dict[str, float] = {}
    for i, name in enumerate(channel_names):
        x = np.nan_to_num(epoch[i])
        dx = np.diff(x)
        ddx = np.diff(dx)
        var_x = float(np.var(x))
        var_dx = float(np.var(dx)) if dx.size else 0.0
        var_ddx = float(np.var(ddx)) if ddx.size else 0.0
        mobility = float(np.sqrt(var_dx / var_x)) if var_x > 0 else 0.0
        mobility_dx = float(np.sqrt(var_ddx / var_dx)) if var_dx > 0 else 0.0
        complexity = float(mobility_dx / mobility) if mobility > 0 else 0.0
        prefix = _safe_name(name)
        features[f"{prefix}_hjorth_activity"] = var_x
        features[f"{prefix}_hjorth_mobility"] = mobility
        features[f"{prefix}_hjorth_complexity"] = complexity
    return features


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name)).strip("_").lower() or "ch"

