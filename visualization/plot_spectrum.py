"""Spectrum and alpha-band plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def plot_spectrum(signals: np.ndarray, sfreq: float, channel_names: Sequence[str], output_path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    signals = np.asarray(signals, dtype=float)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for i, name in enumerate(channel_names):
        freqs, psd = welch(signals[i], fs=sfreq, nperseg=min(signals.shape[1], int(sfreq * 4)))
        ax.semilogy(freqs, psd, linewidth=0.9, label=str(name))
    ax.set_xlim(0, min(60, sfreq / 2))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title("Power spectrum")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_alpha_comparison(signals: np.ndarray, sfreq: float, channel_names: Sequence[str], output_path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    powers = []
    for channel in np.asarray(signals, dtype=float):
        freqs, psd = welch(channel, fs=sfreq, nperseg=min(channel.size, int(sfreq * 4)))
        mask = (freqs >= 8.0) & (freqs <= 13.0)
        powers.append(float(np.trapezoid(psd[mask], freqs[mask])) if np.sum(mask) > 1 else 0.0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.bar(list(channel_names), powers, color=["#2f6f8f", "#b36b30"][: len(powers)])
    ax.set_ylabel("8-13 Hz power")
    ax.set_title("Alpha-band comparison")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
