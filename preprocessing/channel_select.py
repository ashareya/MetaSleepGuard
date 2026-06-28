"""Channel selection helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def select_channels(signals: np.ndarray, channel_names: Sequence[str], wanted: Sequence[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Select channels by name, preserving requested order when available."""

    signals = np.asarray(signals)
    names = list(channel_names)
    if wanted is None:
        return signals, names
    lower_to_index = {name.lower(): i for i, name in enumerate(names)}
    indices: list[int] = []
    selected_names: list[str] = []
    for wanted_name in wanted:
        index = lower_to_index.get(wanted_name.lower())
        if index is not None:
            indices.append(index)
            selected_names.append(names[index])
    if not indices:
        raise ValueError(f"none of requested channels were found: {wanted}")
    return signals[indices, :], selected_names


def choose_two_channels(
    signals: np.ndarray,
    channel_names: Sequence[str],
    preferred: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return two EEG channels, using preferred labels when possible."""

    preferred = preferred or ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "O1", "O2", "EEG1", "EEG2"]
    names = list(channel_names)
    lower_to_index = {name.lower(): i for i, name in enumerate(names)}
    indices: list[int] = []
    for candidate in preferred:
        index = lower_to_index.get(candidate.lower())
        if index is not None and index not in indices:
            indices.append(index)
        if len(indices) == 2:
            break
    if len(indices) < 2:
        indices = list(range(min(2, len(names))))
    if len(indices) < 2:
        raise ValueError("at least two channels are required")
    return np.asarray(signals)[indices, :], [names[i] for i in indices]

