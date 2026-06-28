"""Causal context features.

Only previous epochs from the same subject are used. Future epochs are never
read, so this representation is safe for online use.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def append_causal_context(
    features: np.ndarray,
    subject_ids: Sequence[str],
    history: int = 2,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Concatenate previous ``history`` epochs plus current features."""

    x = np.asarray(features, dtype=float)
    if x.ndim != 2:
        raise ValueError("features must have shape (epochs, features)")
    if history < 0:
        raise ValueError("history must be non-negative")
    subjects = list(subject_ids)
    if len(subjects) != x.shape[0]:
        raise ValueError("subject_ids length must match feature rows")
    output = np.full((x.shape[0], x.shape[1] * (history + 1)), fill_value, dtype=float)
    seen_by_subject: dict[str, list[int]] = {}
    for row_index, subject in enumerate(subjects):
        seen = seen_by_subject.setdefault(str(subject), [])
        block = []
        for offset in range(history, 0, -1):
            source_pos = len(seen) - offset
            block.append(x[seen[source_pos]] if source_pos >= 0 else np.full(x.shape[1], fill_value))
        block.append(x[row_index])
        output[row_index, :] = np.concatenate(block)
        seen.append(row_index)
    return output


def causal_feature_names(feature_names: Sequence[str], history: int = 2) -> list[str]:
    names: list[str] = []
    for offset in range(history, 0, -1):
        names.extend([f"t-{offset}_{name}" for name in feature_names])
    names.extend([f"t0_{name}" for name in feature_names])
    return names

