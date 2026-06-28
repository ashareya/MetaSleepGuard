"""Rule-based artifact detectors for 30-second EEG windows."""

from __future__ import annotations

import numpy as np


def detect_data_dropout(epoch: np.ndarray) -> bool:
    x = np.asarray(epoch, dtype=float)
    if not np.isfinite(x).all():
        return True
    flat_zero = np.abs(x) < 1e-12
    return bool(np.mean(flat_zero) > 0.20)


def detect_flatline(epoch: np.ndarray) -> bool:
    x = np.nan_to_num(np.asarray(epoch, dtype=float))
    for channel in x:
        if np.std(channel) < _flatline_std_threshold(channel):
            return True
        if np.mean(np.abs(np.diff(channel)) < _flatline_diff_threshold(channel)) > 0.98:
            return True
    return False


def detect_abnormal_amplitude(epoch: np.ndarray) -> bool:
    x = np.nan_to_num(np.asarray(epoch, dtype=float))
    threshold = _amplitude_threshold(x)
    return bool(np.max(np.abs(x)) > threshold)


def detect_saturation(epoch: np.ndarray) -> bool:
    x = np.nan_to_num(np.asarray(epoch, dtype=float))
    threshold = _amplitude_threshold(x) * 0.95
    clipped = np.abs(x) >= threshold
    return bool(np.mean(clipped) > 0.01)


def detect_motion_artifact(epoch: np.ndarray) -> bool:
    x = np.nan_to_num(np.asarray(epoch, dtype=float))
    threshold = _amplitude_threshold(x)
    ptp = np.ptp(x, axis=1)
    diff = np.abs(np.diff(x, axis=1))
    return bool(np.any(ptp > threshold) or np.percentile(diff, 99) > threshold * 0.25)


def detect_baseline_drift(epoch: np.ndarray, sfreq: float) -> bool:
    from scipy.signal import welch

    x = np.nan_to_num(np.asarray(epoch, dtype=float))
    for channel in x:
        freqs, psd = welch(channel, fs=sfreq, nperseg=min(channel.size, int(sfreq * 8)))
        low = _band_power(freqs, psd, 0.01, 0.5)
        eeg = _band_power(freqs, psd, 0.5, 35.0) + 1e-24
        if low / eeg > 0.35:
            return True
    return False


def detect_line_noise(epoch: np.ndarray, sfreq: float, line_hz: float = 50.0) -> bool:
    if line_hz >= sfreq / 2:
        return False
    from scipy.signal import welch

    x = np.nan_to_num(np.asarray(epoch, dtype=float))
    for channel in x:
        freqs, psd = welch(channel, fs=sfreq, nperseg=min(channel.size, int(sfreq * 4)))
        line = _band_power(freqs, psd, line_hz - 1.0, line_hz + 1.0)
        eeg = _band_power(freqs, psd, 0.5, min(45.0, sfreq / 2 - 1e-3)) + 1e-24
        if line / eeg > 0.25:
            return True
    return False


def detect_bad_channel(epoch: np.ndarray) -> bool:
    x = np.nan_to_num(np.asarray(epoch, dtype=float))
    if x.shape[0] < 2:
        return False
    stds = np.std(x, axis=1)
    median = np.median(stds) + 1e-24
    imbalance = np.max(stds) / (np.min(stds) + 1e-24) > 8.0
    return bool(imbalance or np.any(stds > median * 8.0) or np.any(stds < median / 8.0))


def detect_bad_channel_names(epoch: np.ndarray, channel_names) -> list[str]:
    """Return channels requiring review due to flatline or variance imbalance."""

    x = np.nan_to_num(np.asarray(epoch, dtype=float))
    names = list(channel_names)
    stds = np.std(x, axis=1)
    if stds.size != len(names):
        raise ValueError("channel_names must match epoch channels")
    median = np.median(stds) + 1e-24
    bad = {
        names[i]
        for i, std in enumerate(stds)
        if std > median * 8.0 or std < median / 8.0 or std < _flatline_std_threshold(x[i])
    }
    if stds.size == 2 and np.max(stds) / (np.min(stds) + 1e-24) > 8.0:
        bad.update(names)
    return sorted(bad)


def _band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if np.sum(mask) < 2:
        return float(np.sum(psd[mask]))
    return float(np.trapezoid(psd[mask], freqs[mask]))


def _amplitude_threshold(x: np.ndarray) -> float:
    # Values around tens usually indicate microvolts; values below 1e-3 usually
    # indicate volts. Pick a threshold in the detected unit without hard-coding a
    # user path or device.
    median_abs = float(np.nanmedian(np.abs(x))) if x.size else 0.0
    return 500.0 if median_abs > 0.01 else 500e-6


def _flatline_std_threshold(channel: np.ndarray) -> float:
    median_abs = float(np.nanmedian(np.abs(channel))) if channel.size else 0.0
    return 0.05 if median_abs > 0.01 else 0.05e-6


def _flatline_diff_threshold(channel: np.ndarray) -> float:
    median_abs = float(np.nanmedian(np.abs(channel))) if channel.size else 0.0
    return 0.01 if median_abs > 0.01 else 0.01e-6
