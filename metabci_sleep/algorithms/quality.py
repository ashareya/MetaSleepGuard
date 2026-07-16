"""Stable public wrapper around MetaSleep-Guard's tested quality audit."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from MetaSleepGuard.quality.quality_audit import QualityResult, audit_epoch, audit_windows


class SleepQualityAuditor:
    """Audit one epoch or a batch without duplicating the underlying rules."""

    def __init__(
        self,
        sfreq: float,
        channel_names: Sequence[str] | None = None,
        epoch_sec: float = 30.0,
    ) -> None:
        if float(sfreq) <= 0:
            raise ValueError("sfreq must be positive")
        if float(epoch_sec) <= 0:
            raise ValueError("epoch_sec must be positive")
        self.sfreq = float(sfreq)
        self.channel_names = None if channel_names is None else list(channel_names)
        self.epoch_sec = float(epoch_sec)

    def audit_epoch(self, epoch: np.ndarray, start_time: float = 0.0) -> QualityResult:
        return audit_epoch(
            epoch,
            self.sfreq,
            channel_names=self.channel_names,
            start_time=start_time,
            epoch_sec=self.epoch_sec,
        )

    def audit_windows(self, signals: np.ndarray) -> list[QualityResult]:
        return audit_windows(
            signals,
            self.sfreq,
            channel_names=self.channel_names,
            epoch_sec=self.epoch_sec,
        )

    def transform(self, epochs: np.ndarray) -> list[dict]:
        epochs = np.asarray(epochs, dtype=float)
        if epochs.ndim != 3:
            raise ValueError("epochs must have shape (epochs, channels, samples)")
        return [
            self.audit_epoch(epoch, start_time=index * self.epoch_sec).to_row()
            for index, epoch in enumerate(epochs)
        ]
