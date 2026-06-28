"""Standard EEG filtering and resampling."""

from __future__ import annotations

import logging
from fractions import Fraction

import numpy as np

LOGGER = logging.getLogger(__name__)


def resample_to(signals: np.ndarray, orig_sfreq: float, target_sfreq: float = 250.0) -> tuple[np.ndarray, float]:
    """Resample ``signals`` to ``target_sfreq`` using polyphase filtering."""

    signals = np.asarray(signals, dtype=float)
    if abs(orig_sfreq - target_sfreq) < 1e-6:
        return signals, float(orig_sfreq)
    try:
        from scipy.signal import resample_poly
    except Exception as exc:  # pragma: no cover - scipy is expected but optional
        raise RuntimeError("scipy is required for resampling") from exc
    ratio = Fraction(float(target_sfreq) / float(orig_sfreq)).limit_denominator(1000)
    LOGGER.info("Resampling from %.3f Hz to %.3f Hz", orig_sfreq, target_sfreq)
    return resample_poly(signals, ratio.numerator, ratio.denominator, axis=-1), float(target_sfreq)


def bandpass_filter(signals: np.ndarray, sfreq: float, low_hz: float = 0.3, high_hz: float = 35.0, order: int = 4) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filtering."""

    from scipy.signal import butter, sosfiltfilt

    signals = np.asarray(signals, dtype=float)
    nyq = sfreq / 2.0
    high = min(high_hz, nyq * 0.95)
    if low_hz <= 0 or low_hz >= high:
        raise ValueError("invalid bandpass cutoff")
    sos = butter(order, [low_hz / nyq, high / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signals, axis=-1)


def notch_filter(signals: np.ndarray, sfreq: float, notch_hz: float = 50.0, quality: float = 30.0) -> np.ndarray:
    """Apply a 50 Hz notch filter when the frequency is below Nyquist."""

    if notch_hz >= sfreq / 2.0:
        return np.asarray(signals, dtype=float)
    from scipy.signal import filtfilt, iirnotch

    b, a = iirnotch(notch_hz / (sfreq / 2.0), quality)
    return filtfilt(b, a, np.asarray(signals, dtype=float), axis=-1)


def preprocess_signal(
    signals: np.ndarray,
    sfreq: float,
    target_sfreq: float = 250.0,
    bandpass: tuple[float, float] = (0.3, 35.0),
    notch_hz: float | None = 50.0,
) -> tuple[np.ndarray, float]:
    """Resample, bandpass, and notch filter an EEG recording."""

    processed, new_sfreq = resample_to(signals, sfreq, target_sfreq)
    processed = bandpass_filter(processed, new_sfreq, bandpass[0], bandpass[1])
    if notch_hz is not None:
        processed = notch_filter(processed, new_sfreq, notch_hz)
    return processed, new_sfreq

