"""Frequency-domain bandpower features."""

from __future__ import annotations

from typing import Sequence

import numpy as np

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}


def bandpower_features(epoch: np.ndarray, sfreq: float, channel_names: Sequence[str]) -> dict[str, float]:
    from scipy.signal import welch

    epoch = np.asarray(epoch, dtype=float)
    features: dict[str, float] = {}
    nperseg = min(epoch.shape[-1], int(round(sfreq * 4)))
    for i, name in enumerate(channel_names):
        x = np.nan_to_num(epoch[i])
        freqs, psd = welch(x, fs=sfreq, nperseg=max(8, nperseg))
        total_mask = (freqs >= 0.5) & (freqs <= 35.0)
        total_power = _integrate(freqs[total_mask], psd[total_mask]) + 1e-24
        prefix = _safe_name(name)
        for band, (low, high) in BANDS.items():
            mask = (freqs >= low) & (freqs < high)
            power = _integrate(freqs[mask], psd[mask])
            features[f"{prefix}_{band}_power"] = float(power)
            features[f"{prefix}_{band}_rel_power"] = float(power / total_power)
    return features


def _integrate(freqs: np.ndarray, psd: np.ndarray) -> float:
    if freqs.size < 2:
        return float(np.sum(psd))
    return float(np.trapezoid(psd, freqs))


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name)).strip("_").lower() or "ch"
