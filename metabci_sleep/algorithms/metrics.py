"""Transparent engineering sleep-structure metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from MetaSleepGuard.preprocessing.label_mapping import map_raw_stage


@dataclass
class SleepMetrics:
    epoch_sec: float = 30.0

    def compute(self, stages: Sequence[str]) -> dict:
        if self.epoch_sec <= 0:
            raise ValueError("epoch_sec must be positive")
        normalized = [self._normalize(stage) for stage in stages]
        n = len(normalized)
        counts = {stage: normalized.count(stage) for stage in ["W", "N1", "N2", "N3", "REM"]}
        unknown = sum(stage in {"UNKNOWN", "暂不判定"} for stage in normalized)
        sleep_indices = [i for i, stage in enumerate(normalized) if stage in {"N1", "N2", "N3", "REM"}]
        total_min = n * self.epoch_sec / 60.0
        sleep_min = len(sleep_indices) * self.epoch_sec / 60.0
        efficiency = sleep_min / total_min if total_min else 0.0
        if sleep_indices:
            first_sleep, last_sleep = sleep_indices[0], sleep_indices[-1]
            latency_min = first_sleep * self.epoch_sec / 60.0
            waso_epochs = sum(normalized[i] == "W" for i in range(first_sleep + 1, last_sleep + 1))
            awakenings = self._wake_runs(normalized[first_sleep + 1 : last_sleep + 1])
        else:
            latency_min = total_min
            waso_epochs = 0
            awakenings = 0
        waso_min = waso_epochs * self.epoch_sec / 60.0
        stage_minutes = {stage: count * self.epoch_sec / 60.0 for stage, count in counts.items()}
        sleep_ratios = {
            stage: (stage_minutes[stage] / sleep_min if sleep_min else 0.0)
            for stage in ["N1", "N2", "N3", "REM"]
        }
        score = self._engineering_score(efficiency, latency_min, waso_min, sleep_min, sleep_ratios)
        return {
            "recording_minutes": total_min,
            "total_sleep_minutes": sleep_min,
            "sleep_efficiency": efficiency,
            "sleep_onset_latency_minutes": latency_min,
            "waso_minutes": waso_min,
            "awakenings_after_sleep_onset": awakenings,
            "stage_minutes": stage_minutes,
            "sleep_stage_ratios": sleep_ratios,
            "unknown_or_rejected_epochs": unknown,
            "engineering_sleep_score": score,
            "score_disclaimer": "Engineering heuristic only; not a clinical scale or medical diagnosis.",
        }

    @staticmethod
    def _normalize(stage: str) -> str:
        text = str(stage).strip().upper()
        if text == "暂不判定":
            return text
        if text == "NREM":
            return "N2"
        if text == "LIGHT":
            return "N2"
        return map_raw_stage(text)

    @staticmethod
    def _wake_runs(stages: Sequence[str]) -> int:
        runs = 0
        previous = None
        for stage in stages:
            if stage == "W" and previous != "W":
                runs += 1
            previous = stage
        return runs

    @staticmethod
    def _engineering_score(efficiency, latency_min, waso_min, sleep_min, ratios) -> float:
        score = 100.0
        score -= min(30.0, max(0.0, 0.85 - efficiency) * 100.0)
        score -= min(20.0, max(0.0, latency_min - 20.0) * 0.5)
        score -= min(25.0, (waso_min / sleep_min * 100.0) if sleep_min else 25.0)
        restorative = ratios.get("N3", 0.0) + ratios.get("REM", 0.0)
        score -= min(15.0, max(0.0, 0.30 - restorative) * 50.0)
        return round(max(0.0, min(100.0, score)), 1)
