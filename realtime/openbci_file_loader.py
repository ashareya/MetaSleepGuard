"""File replay loaders for OpenBCI CSV/TXT and MNE-supported EEG files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..preprocessing.channel_select import choose_two_channels


@dataclass
class ReplayData:
    signals: np.ndarray
    sfreq: float
    channel_names: list[str]
    metadata: dict


def load_openbci_csv(
    path: str | Path,
    channel_indices: Sequence[int] = (1, 6),
    sfreq: float = 250.0,
    units: str = "microvolts",
) -> ReplayData:
    """Load OpenBCI GUI CSV/TXT exports.

    ``channel_indices`` are zero-based EXG indices, so ``(1, 6)`` means Ch2 and
    Ch7 in OpenBCI GUI naming.
    """

    path = Path(path)
    df = pd.read_csv(path, comment="%", engine="python")
    df = df.dropna(axis=1, how="all")
    exg_cols = [col for col in df.columns if "EXG" in str(col).upper() or "CHANNEL" in str(col).upper()]
    if not exg_cols:
        numeric = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        exg_cols = numeric[1:] if len(numeric) > 2 else numeric
    selected_cols = []
    for index in channel_indices:
        if index < len(exg_cols):
            selected_cols.append(exg_cols[index])
    if not selected_cols:
        selected_cols = exg_cols[:2]
    if len(selected_cols) < 2:
        raise ValueError("OpenBCI file must contain at least two EEG channels")
    signals = df[selected_cols].apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both").to_numpy().T
    unit_key = units.lower().replace("µ", "u")
    if unit_key in {"microvolt", "microvolts", "uv"}:
        signals = signals * 1e-6
    elif unit_key not in {"volt", "volts", "v"}:
        raise ValueError(f"unsupported OpenBCI signal units: {units}")
    return ReplayData(
        signals=np.asarray(signals, dtype=float),
        sfreq=float(sfreq),
        channel_names=[str(col) for col in selected_cols],
        metadata={"path": str(path), "source": "openbci_csv", "input_units": units, "output_units": "volts"},
    )


def load_mne_file(path: str | Path, channels: Sequence[str] | None = None) -> ReplayData:
    try:
        import mne
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("mne is required to read BDF/FIF/EDF files") from exc

    path = Path(path)
    readers = {
        ".bdf": mne.io.read_raw_bdf,
        ".fif": mne.io.read_raw_fif,
        ".edf": mne.io.read_raw_edf,
    }
    try:
        reader = readers[path.suffix.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported MNE replay file type: {path.suffix.lower()}") from exc
    raw = reader(path, preload=False, verbose="ERROR")
    if channels:
        available = [ch for ch in channels if ch in raw.ch_names]
        if len(available) != 2:
            raise ValueError(f"exactly two requested channels must exist; found {available}")
        raw.pick(available)
    else:
        empty = np.empty((len(raw.ch_names), 0), dtype=float)
        _, selected_names = choose_two_channels(empty, raw.ch_names)
        raw.pick(selected_names)
    return ReplayData(
        signals=raw.get_data(),
        sfreq=float(raw.info["sfreq"]),
        channel_names=list(raw.ch_names),
        metadata={"path": str(path), "source": path.suffix.lower().lstrip(".")},
    )


def load_replay_file(path: str | Path, **kwargs) -> ReplayData:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return load_openbci_csv(
            path,
            channel_indices=kwargs.get("channel_indices", (1, 6)),
            sfreq=kwargs.get("sfreq", 250.0),
            units=kwargs.get("units", "microvolts"),
        )
    if suffix in {".bdf", ".fif", ".edf"}:
        return load_mne_file(path, channels=kwargs.get("channels"))
    raise ValueError(f"unsupported replay file type: {suffix}")
