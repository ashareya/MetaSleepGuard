"""Adapter between project OpenBCI flows and MetaBCI BrainFlow primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
from typing import Sequence

from .metabci_component_check import inspect_metabci_components


@dataclass(frozen=True)
class BrainFlowAdapterResult:
    ok: bool
    metabci_brainflow_available: bool
    ring_buffer_full: bool
    ring_buffer_items: list[int]
    synthetic_shape: tuple[int, int]
    sfreq: float
    channel_names: list[str]
    project_stream: str
    metabci_entrypoints: list[str]
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _brainflow_entrypoints() -> list[str]:
    modules = [
        "metabci.brainflow",
        "metabci.brainflow.amplifiers",
        "metabci.brainflow.workers",
        "metabci.brainflow.logger",
    ]
    imported = []
    for module_name in modules:
        importlib.import_module(module_name)
        imported.append(module_name)
    return imported


def verify_metabci_ring_buffer(size: int = 4) -> dict:
    """Exercise MetaBCI's own BrainFlow RingBuffer implementation."""

    amplifiers = importlib.import_module("metabci.brainflow.amplifiers")
    ring_buffer = amplifiers.RingBuffer(size=size)
    for value in range(size):
        ring_buffer.append(value)
    return {
        "class": "metabci.brainflow.amplifiers.RingBuffer",
        "is_full": bool(ring_buffer.isfull()),
        "items": list(ring_buffer.get_all()),
    }


def describe_openbci_runtime_alignment(
    channel_indices: Sequence[int] = (1, 6),
    sfreq: float = 250.0,
) -> dict:
    status = inspect_metabci_components()["brainflow"]
    return {
        "metabci_component": status.to_dict(),
        "project_runtime_streams": [
            "MetaSleepGuard.realtime.openbci_brainflow_stream.OpenBCICytonStream",
            "MetaSleepGuard.realtime.openbci_brainflow_stream.SyntheticBrainFlowStream",
            "MetaSleepGuard.realtime.openbci_file_loader.load_replay_file",
        ],
        "openbci_mapping": {
            "board": "OpenBCI Cyton via BrainFlow BoardShim when hardware is present",
            "channel_indices_zero_based": list(channel_indices),
            "default_sampling_rate_hz": float(sfreq),
            "window_seconds": 30,
        },
        "metabci_brainflow_usage": [
            "import metabci.brainflow and child modules during status/integration tests",
            "use metabci.brainflow.amplifiers.RingBuffer for lightweight online-buffer smoke checks",
            "align OpenBCI chunks with MetaBCI BrainFlow's acquisition/worker/ring-buffer responsibilities",
        ],
    }


def run_brainflow_adapter_smoke() -> BrainFlowAdapterResult:
    status = inspect_metabci_components()["brainflow"]
    if not status.available:
        return BrainFlowAdapterResult(
            ok=False,
            metabci_brainflow_available=False,
            ring_buffer_full=False,
            ring_buffer_items=[],
            synthetic_shape=(0, 0),
            sfreq=0.0,
            channel_names=[],
            project_stream="",
            metabci_entrypoints=[],
            notes=[status.error or status.detail],
        )

    entrypoints = _brainflow_entrypoints()
    ring = verify_metabci_ring_buffer()

    from MetaSleepGuard.realtime.openbci_brainflow_stream import SyntheticBrainFlowStream

    stream = SyntheticBrainFlowStream()
    stream.start()
    try:
        chunk = stream.read(seconds=0.2)
    finally:
        stream.stop()

    ok = bool(ring["is_full"]) and tuple(chunk.data.shape) == (2, 50)
    return BrainFlowAdapterResult(
        ok=ok,
        metabci_brainflow_available=True,
        ring_buffer_full=bool(ring["is_full"]),
        ring_buffer_items=list(ring["items"]),
        synthetic_shape=tuple(int(value) for value in chunk.data.shape),
        sfreq=float(chunk.sfreq),
        channel_names=list(chunk.channel_names),
        project_stream="SyntheticBrainFlowStream",
        metabci_entrypoints=entrypoints,
        notes=[
            "MetaBCI BrainFlow was imported from the local metabci environment.",
            "MetaBCI RingBuffer was exercised without OpenBCI hardware.",
            "Project OpenBCI replay/realtime streams remain intact and are aligned to this acquisition layer.",
        ],
    )
