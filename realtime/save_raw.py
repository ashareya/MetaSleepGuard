"""Raw OpenBCI data persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


class RawCsvWriter:
    """Append timestamped multi-channel samples to a CSV without buffering a full night."""

    def __init__(self, output_path: str | Path, sfreq: float, channel_names: Sequence[str]) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.sfreq = float(sfreq)
        self.channel_names = list(channel_names)
        self.samples_written = 0
        pd.DataFrame(columns=["time_sec", *self.channel_names]).to_csv(self.output_path, index=False)

    def append(self, signals: np.ndarray) -> None:
        signals = np.asarray(signals, dtype=float)
        if signals.ndim != 2 or signals.shape[0] != len(self.channel_names):
            raise ValueError("signals must have shape (configured channels, samples)")
        if signals.shape[1] == 0:
            return
        time = (self.samples_written + np.arange(signals.shape[1])) / self.sfreq
        frame = {"time_sec": time}
        for i, name in enumerate(self.channel_names):
            frame[name] = signals[i]
        pd.DataFrame(frame).to_csv(self.output_path, mode="a", header=False, index=False)
        self.samples_written += signals.shape[1]


def save_raw_csv(
    signals: np.ndarray,
    sfreq: float,
    channel_names: Sequence[str],
    output_path: str | Path,
    start_time: float = 0.0,
) -> Path:
    signals = np.asarray(signals, dtype=float)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time = start_time + np.arange(signals.shape[1]) / float(sfreq)
    data = {"time_sec": time}
    for i, name in enumerate(channel_names):
        data[str(name)] = signals[i]
    pd.DataFrame(data).to_csv(output_path, index=False)
    return output_path
