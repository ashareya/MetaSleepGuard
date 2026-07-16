"""Brainstim-aligned sleep quality calibration protocol."""

from __future__ import annotations

import importlib
from pathlib import Path


class SleepCalibrationProtocol:
    def metabci_brainstim_available(self) -> bool:
        try:
            importlib.import_module("metabci.brainstim")
            return True
        except Exception:
            return False

    def event_plan(self) -> list[dict]:
        from MetaSleepGuard.brainstim_task.calibration_task import DEFAULT_EVENTS

        return [
            {"marker": event.marker, "prompt": event.prompt, "duration_sec": event.duration_sec}
            for event in DEFAULT_EVENTS
        ]

    def run(
        self,
        output_csv: str | Path,
        dry_run: bool = False,
        use_psychopy: bool = True,
        countdown_sec: int = 3,
    ) -> Path:
        from MetaSleepGuard.brainstim_task.calibration_task import run_calibration_task

        return run_calibration_task(
            output_csv,
            countdown_sec=countdown_sec,
            dry_run=dry_run,
            use_psychopy=use_psychopy,
        )
