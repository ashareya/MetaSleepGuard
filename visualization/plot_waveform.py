"""Waveform plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def plot_waveform(signals: np.ndarray, sfreq: float, channel_names: Sequence[str], output_path: str | Path, seconds: float = 30.0) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    signals = np.asarray(signals, dtype=float)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = min(signals.shape[1], int(round(sfreq * seconds)))
    t = np.arange(n) / float(sfreq)
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    offset = 0.0
    for i, name in enumerate(channel_names):
        y = signals[i, :n]
        ax.plot(t, y + offset, linewidth=0.8, label=str(name))
        offset += np.nanstd(y) * 6 if np.nanstd(y) > 0 else 1.0
    ax.set_xlabel("Time (s)")
    ax.set_title("Two-channel EEG waveform")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path

