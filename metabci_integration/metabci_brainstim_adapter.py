"""Adapter between project calibration markers and MetaBCI Brainstim availability."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import importlib
from pathlib import Path

from .metabci_component_check import ComponentStatus, inspect_metabci_components


@dataclass(frozen=True)
class BrainstimAdapterResult:
    ok: bool
    metabci_brainstim_available: bool
    marker_csv: str
    marker_rows: int
    marker_names: list[str]
    component: dict
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def probe_metabci_brainstim() -> ComponentStatus:
    return inspect_metabci_components()["brainstim"]


def _brainstim_entrypoints_if_available() -> list[str]:
    modules = [
        "metabci.brainstim",
        "metabci.brainstim.framework",
        "metabci.brainstim.paradigm",
        "metabci.brainstim.utils",
    ]
    imported = []
    for module_name in modules:
        importlib.import_module(module_name)
        imported.append(module_name)
    return imported


def build_brainstim_marker_plan() -> dict:
    from MetaSleepGuard.brainstim_task.calibration_task import DEFAULT_EVENTS

    status = probe_metabci_brainstim()
    entrypoints: list[str] = []
    if status.available:
        entrypoints = _brainstim_entrypoints_if_available()
    return {
        "metabci_component": status.to_dict(),
        "metabci_entrypoints": entrypoints,
        "project_calibration_task": "MetaSleepGuard.brainstim_task.calibration_task.run_calibration_task",
        "project_marker_outlet": "MetaSleepGuard.brainstim_task.lsl_marker.MarkerOutlet",
        "marker_sequence": [
            {
                "marker": event.marker,
                "duration_sec": float(event.duration_sec),
            }
            for event in DEFAULT_EVENTS
        ],
        "alignment": [
            "MetaBCI Brainstim is the checked stimulation/paradigm platform boundary.",
            "The project calibration task emits LSL-compatible markers and CSV logs for every prompt.",
            "If metabci.brainstim is blocked by PsychoPy in the analysis env, the adapter records the exact import error and still validates marker logging.",
        ],
    }


def run_brainstim_adapter_smoke(output_dir: str | Path) -> BrainstimAdapterResult:
    from MetaSleepGuard.brainstim_task.calibration_task import TaskEvent, run_calibration_task

    status = probe_metabci_brainstim()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    marker_csv = output_dir / "brainstim_marker_smoke.csv"
    event = TaskEvent(
        marker="metabci_brainstim_alignment",
        prompt="MetaBCI Brainstim marker alignment check",
        duration_sec=0.0,
    )
    run_calibration_task(
        marker_csv,
        events=[event],
        countdown_sec=0,
        dry_run=True,
        use_psychopy=False,
    )
    with marker_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    markers = [row["marker"] for row in rows]
    ok = markers == [event.marker]
    notes = [
        "Dry-run calibration wrote a marker-aligned CSV log through the project Brainstim adapter.",
        "This smoke does not require a visual PsychoPy window or OpenBCI hardware.",
    ]
    if status.available:
        notes.append("metabci.brainstim is importable in this Python environment.")
    else:
        notes.append(status.error or status.detail)
    return BrainstimAdapterResult(
        ok=ok,
        metabci_brainstim_available=status.available,
        marker_csv=str(marker_csv),
        marker_rows=len(rows),
        marker_names=markers,
        component=status.to_dict(),
        notes=notes,
    )
