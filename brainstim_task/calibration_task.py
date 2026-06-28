"""Brainstim/PsychoPy calibration paradigm for alpha and artifact checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import time
from typing import Sequence

from .lsl_marker import MarkerOutlet


@dataclass
class TaskEvent:
    marker: str
    prompt: str
    duration_sec: float


DEFAULT_EVENTS = [
    TaskEvent("eyes_open_adapt", "睁眼适应，保持放松", 60),
    TaskEvent("eyes_open_rest_1", "睁眼静息，请注视屏幕中央", 120),
    TaskEvent("eyes_closed_rest_1", "闭眼静息，保持清醒放松", 120),
    TaskEvent("eyes_open_rest_2", "睁眼静息，请减少眨眼", 120),
    TaskEvent("eyes_closed_rest_2", "闭眼静息，保持身体稳定", 120),
    TaskEvent("blink", "请连续眨眼", 15),
    TaskEvent("clench_teeth", "请轻咬牙，随后放松", 15),
    TaskEvent("turn_head", "请缓慢左右转头", 15),
    TaskEvent("move_cable", "请轻移动电极线，制造接触伪迹", 15),
]


def run_calibration_task(
    output_csv: str | Path,
    events: Sequence[TaskEvent] = DEFAULT_EVENTS,
    countdown_sec: int = 3,
    dry_run: bool = False,
    use_psychopy: bool = True,
) -> Path:
    """Run the calibration task and save marker-aligned event logs."""

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    outlet = MarkerOutlet()
    window = _make_window() if use_psychopy and not dry_run else None
    try:
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["marker", "prompt", "lsl_timestamp", "unix_timestamp", "duration_sec"],
            )
            writer.writeheader()
            for event in events:
                _countdown(window, countdown_sec, dry_run)
                _show_prompt(window, event.prompt)
                lsl_timestamp = outlet.push(event.marker)
                writer.writerow(
                    {
                        "marker": event.marker,
                        "prompt": event.prompt,
                        "lsl_timestamp": lsl_timestamp,
                        "unix_timestamp": outlet.last_wall_time,
                        "duration_sec": event.duration_sec,
                    }
                )
                handle.flush()
                if not dry_run:
                    time.sleep(event.duration_sec)
    finally:
        if window is not None:
            window.close()
    return output_csv


def _make_window():
    try:
        from psychopy import visual
    except Exception as exc:  # pragma: no cover - optional stim environment
        raise RuntimeError("PsychoPy is required unless dry_run=True or use_psychopy=False") from exc
    return visual.Window(size=(800, 600), color="black", units="height")


def _show_prompt(window, text: str) -> None:
    if window is None:
        print(text)
        return
    from psychopy import visual

    message = visual.TextStim(window, text=text, color="white", height=0.06, wrapWidth=1.2)
    message.draw()
    window.flip()


def _countdown(window, seconds: int, dry_run: bool) -> None:
    for remaining in range(seconds, 0, -1):
        _show_prompt(window, f"{remaining}")
        if not dry_run:
            time.sleep(1.0)
