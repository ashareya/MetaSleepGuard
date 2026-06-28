"""Thirty-second epoching, label alignment, and subject-level splitting."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np


def epoch_signal(signals: np.ndarray, sfreq: float, epoch_sec: float = 30.0) -> np.ndarray:
    """Slice signals into shape ``(epochs, channels, samples_per_epoch)``."""

    signals = np.asarray(signals, dtype=float)
    if signals.ndim != 2:
        raise ValueError("signals must have shape (channels, samples)")
    samples_per_epoch = int(round(float(sfreq) * float(epoch_sec)))
    if samples_per_epoch <= 0:
        raise ValueError("samples_per_epoch must be positive")
    n_epochs = signals.shape[1] // samples_per_epoch
    trimmed = signals[:, : n_epochs * samples_per_epoch]
    if n_epochs == 0:
        return np.empty((0, signals.shape[0], samples_per_epoch), dtype=float)
    return trimmed.reshape(signals.shape[0], n_epochs, samples_per_epoch).transpose(1, 0, 2)


def align_epoch_labels(labels: Sequence[str], n_epochs: int) -> list[str]:
    """Align label count to signal epoch count by truncating or padding unknowns."""

    aligned = list(labels[:n_epochs])
    if len(aligned) < n_epochs:
        aligned.extend(["UNKNOWN"] * (n_epochs - len(aligned)))
    return aligned


def subject_level_split(
    subject_ids: Sequence[str],
    test_fraction: float = 0.2,
    seed: int = 13,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train/test boolean masks with no subject leakage."""

    subject_ids = np.asarray(subject_ids)
    unique = np.unique(subject_ids)
    if unique.size < 2:
        raise ValueError("at least two subjects are required for subject-level split")
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    n_test = max(1, int(round(unique.size * test_fraction)))
    test_subjects = set(shuffled[:n_test])
    test_mask = np.array([subject in test_subjects for subject in subject_ids], dtype=bool)
    return ~test_mask, test_mask


def leave_one_subject_indices(subject_ids: Sequence[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return train/test masks for each leave-one-subject split."""

    subject_ids = np.asarray(subject_ids)
    splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for subject in np.unique(subject_ids):
        test_mask = subject_ids == subject
        splits[str(subject)] = (~test_mask, test_mask)
    return splits


def group_epoch_indices(subject_ids: Sequence[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, subject in enumerate(subject_ids):
        groups[str(subject)].append(i)
    return dict(groups)

