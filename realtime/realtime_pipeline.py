"""Real-time and replay pipeline for 30-second quality and staging."""

from __future__ import annotations

from pathlib import Path
import csv
import logging
from typing import Sequence

import numpy as np

from ..models.sleep_inference import SleepInference
from ..preprocessing.filters import preprocess_signal
from ..quality.quality_audit import audit_epoch
from .ring_buffer import RingBuffer

LOGGER = logging.getLogger(__name__)


class RealtimePipeline:
    def __init__(
        self,
        sfreq: float = 250.0,
        channel_names: Sequence[str] = ("Ch2", "Ch7"),
        model_bundle: dict | str | Path | None = None,
        epoch_sec: float = 30.0,
        buffer_minutes: float = 10.0,
        output_log: str | Path | None = None,
    ) -> None:
        if sfreq <= 0:
            raise ValueError("sfreq must be positive")
        if epoch_sec <= 0:
            raise ValueError("epoch_sec must be positive")
        if buffer_minutes <= 0 or buffer_minutes * 60.0 < epoch_sec:
            raise ValueError("buffer_minutes must hold at least one complete epoch")
        if not channel_names:
            raise ValueError("at least one channel name is required")
        self.sfreq = float(sfreq)
        self.channel_names = list(channel_names)
        self.epoch_sec = float(epoch_sec)
        self.samples_per_epoch = int(round(self.sfreq * self.epoch_sec))
        self.buffer = RingBuffer(len(self.channel_names), int(round(self.sfreq * 60.0 * buffer_minutes)))
        self.inference = SleepInference(model_bundle) if model_bundle is not None else None
        self.last_processed_sample = 0
        self.output_log = Path(output_log) if output_log else None
        if self.output_log:
            self.output_log.parent.mkdir(parents=True, exist_ok=True)
            with self.output_log.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=_LOG_FIELDS)
                writer.writeheader()

    def append_and_process(self, samples: np.ndarray) -> list[dict]:
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2 or samples.shape[0] != len(self.channel_names):
            raise ValueError("samples must have shape (configured channels, samples)")
        outputs: list[dict] = []
        cursor = 0
        while cursor < samples.shape[1]:
            pending = self.buffer.samples_seen - self.last_processed_sample
            needed = self.samples_per_epoch - pending
            take = min(needed, samples.shape[1] - cursor)
            self.buffer.append(samples[:, cursor : cursor + take])
            cursor += take
            if self.buffer.samples_seen - self.last_processed_sample == self.samples_per_epoch:
                epoch = self.buffer.get_last(self.samples_per_epoch)
                start = self.last_processed_sample / self.sfreq
                outputs.append(self.process_epoch(epoch, start_time=start))
                self.last_processed_sample += self.samples_per_epoch
        return outputs

    def process_epoch(self, epoch: np.ndarray, start_time: float = 0.0) -> dict:
        quality = audit_epoch(epoch, self.sfreq, self.channel_names, start_time=start_time, epoch_sec=self.epoch_sec)
        processed, new_sfreq = preprocess_signal(epoch, self.sfreq)
        if self.inference is None:
            row = {
                "window_start_time": quality.window_start_time,
                "window_end_time": quality.window_end_time,
                "stage": "暂不判定",
                "confidence": 0.0,
                "accepted": False,
                "reason": "no_model_loaded",
                "quality_grade": quality.quality_grade,
                "quality_score": quality.quality_score,
                "bad_flags": "|".join(quality.bad_flags),
            }
        else:
            result = self.inference.predict_epoch(processed, new_sfreq, self.channel_names, quality)
            row = {
                "window_start_time": quality.window_start_time,
                "window_end_time": quality.window_end_time,
                "stage": result["stage"],
                "confidence": result["confidence"],
                "accepted": result["accepted"],
                "reason": result["reason"],
                "quality_grade": quality.quality_grade,
                "quality_score": quality.quality_score,
                "bad_flags": "|".join(quality.bad_flags),
            }
        self._write_row(row)
        LOGGER.info("Realtime window %.1f-%.1f: %s", row["window_start_time"], row["window_end_time"], row["reason"])
        return row

    def _write_row(self, row: dict) -> None:
        if not self.output_log:
            return
        with self.output_log.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_LOG_FIELDS)
            writer.writerow(row)


def replay_array(signals: np.ndarray, sfreq: float, channel_names: Sequence[str], **pipeline_kwargs) -> list[dict]:
    pipeline = RealtimePipeline(sfreq=sfreq, channel_names=channel_names, **pipeline_kwargs)
    signals = np.asarray(signals, dtype=float)
    if signals.ndim != 2:
        raise ValueError("signals must have shape (channels, samples)")
    if signals.shape[0] != len(channel_names):
        raise ValueError("channel_names must match the signal channel count")

    outputs: list[dict] = []
    chunk_samples = max(1, int(round(float(sfreq))))
    for start in range(0, signals.shape[1], chunk_samples):
        outputs.extend(pipeline.append_and_process(signals[:, start : start + chunk_samples]))
    return outputs


_LOG_FIELDS = [
    "window_start_time",
    "window_end_time",
    "stage",
    "confidence",
    "accepted",
    "reason",
    "quality_grade",
    "quality_score",
    "bad_flags",
]
