"""Generic timestamp-based 30-second window integrity auditing."""

from __future__ import annotations

import numpy as np

from MetaSleepGuard.quality.quality_audit import audit_epoch


class WindowIntegrityAuditor:
    def __init__(self, sfreq: float = 250.0, epoch_sec: float = 30.0) -> None:
        if sfreq <= 0 or epoch_sec <= 0:
            raise ValueError("sfreq and epoch_sec must be positive")
        self.sfreq = float(sfreq)
        self.epoch_sec = float(epoch_sec)

    def audit(
        self,
        timestamps: np.ndarray,
        signals_uv: np.ndarray,
        start_time: float | None = None,
    ) -> list[dict]:
        timestamps = np.asarray(timestamps, dtype=float)
        signals_uv = np.asarray(signals_uv, dtype=float)
        if timestamps.ndim != 1 or signals_uv.ndim != 2:
            raise ValueError("timestamps must be 1D and signals_uv must be channels x samples")
        if signals_uv.shape[1] != len(timestamps):
            raise ValueError("timestamp and sample counts must match")
        if len(timestamps) == 0:
            return []
        if signals_uv.shape[0] != 2:
            raise ValueError("current OpenBCI quality rules require exactly two channels")
        origin = float(timestamps[0] if start_time is None else start_time)
        final = float(timestamps[-1] + 1.0 / self.sfreq)
        n_windows = int(np.ceil(max(0.0, final - origin) / self.epoch_sec))
        rows = []
        for index in range(n_windows):
            start = origin + index * self.epoch_sec
            end = start + self.epoch_sec
            mask = (timestamps >= start) & (timestamps < end)
            selected_times = timestamps[mask]
            selected = signals_uv[:, mask]
            expected = int(round(self.sfreq * self.epoch_sec))
            sample_ratio = min(1.0, selected.shape[1] / expected)
            if len(selected_times):
                covered = min(self.epoch_sec, max(0.0, selected_times[-1] - selected_times[0] + 1.0 / self.sfreq))
            else:
                covered = 0.0
            coverage = covered / self.epoch_sec
            if selected.shape[1] < 8:
                flags = ["data_dropout"]
                grade = "D"
                score = 30.0
                reliable = False
            else:
                quality_result = audit_epoch(
                    selected * 1e-6,
                    self.sfreq,
                    channel_names=["Ch2", "Ch7"],
                    start_time=start,
                    epoch_sec=self.epoch_sec,
                )
                flags = list(quality_result.bad_flags)
                grade = quality_result.quality_grade
                score = quality_result.quality_score
                reliable = quality_result.is_reliable
            if min(coverage, sample_ratio) < 0.8:
                flags.append("effective_data_ratio_insufficient")
                grade = "D"
                score = min(score, 30.0)
                reliable = False
            quality = {
                "quality_grade": grade,
                "quality_score": score,
                "quality_flags": "|".join(sorted(set(flags))),
                "usable_for_window_inference": bool(reliable),
                "trusted_output": "可进入模型" if reliable else "暂不判定",
            }
            rows.append(
                {
                    "window_index": index,
                    "window_start_time": start,
                    "window_end_time": end,
                    "sample_count": int(selected.shape[1]),
                    "expected_samples": expected,
                    "coverage_ratio": coverage,
                    "missing_duration_sec": self.epoch_sec - covered,
                    "sample_completeness_ratio": sample_ratio,
                    "effective_data_ratio": min(coverage, sample_ratio),
                    **quality,
                }
            )
        return rows
