"""Continuous 30-second sleep staging through MetaBCI's paradigm contract."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

try:
    from metabci.brainda.datasets.base import BaseDataset
    from metabci.brainda.paradigms.base import BaseParadigm
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "SleepStaging requires MetaBCI Brainda. Install metabci-sleep with its dependencies."
    ) from exc

from MetaSleepGuard.preprocessing.label_mapping import (
    map_stage_for_task,
    normalize_task,
    task_classes,
)
from metabci_sleep.datasets.sleep_edf import STAGE_EVENT_IDS


EVENT_ID_TO_STAGE = {event_id: stage for stage, event_id in STAGE_EVENT_IDS.items()}


class SleepStaging(BaseParadigm):
    """Extract continuous sleep epochs from a compatible MetaBCI dataset."""

    def __init__(
        self,
        task: str = "5class",
        channels: Optional[list[str]] = None,
        srate: Optional[float] = None,
        epoch_sec: float = 30.0,
    ) -> None:
        self.task = normalize_task(task)
        if self.task not in {"3class", "4class", "5class"}:
            raise ValueError("SleepStaging supports task='3class', '4class', or '5class'")
        if float(epoch_sec) != 30.0:
            raise ValueError("SleepStaging currently supports 30-second windows only")
        self.epoch_sec = float(epoch_sec)
        super().__init__(channels=channels, events=None, intervals=[(0.0, self.epoch_sec)], srate=srate)

    def is_valid(self, dataset: BaseDataset) -> bool:
        return isinstance(dataset, BaseDataset) and getattr(dataset, "paradigm", None) == "sleep_staging"

    def get_data(
        self,
        dataset: BaseDataset,
        subjects: Optional[list[Union[int, str]]] = None,
        label_encode: bool = True,
        return_concat: bool = True,
        n_jobs: int = 1,
        verbose: Optional[bool] = None,
    ):
        """Return ``X, y, meta`` using MetaBCI-compatible shapes and metadata."""

        del n_jobs, verbose
        if not self.is_valid(dataset):
            raise TypeError("dataset must be a MetaBCI BaseDataset with paradigm='sleep_staging'")
        selected_subjects = list(dataset.subjects if subjects is None else subjects)
        if not selected_subjects:
            raise ValueError("at least one subject is required")
        invalid = [subject for subject in selected_subjects if subject not in dataset.subjects]
        if invalid:
            raise ValueError(f"invalid subjects: {invalid}")

        event_arrays: dict[str, list[np.ndarray]] = {stage: [] for stage in task_classes(self.task)}
        event_meta: dict[str, list[dict]] = {stage: [] for stage in task_classes(self.task)}
        selected_channels: list[str] | None = None

        nested = dataset.get_data(selected_subjects)
        for subject, sessions in nested.items():
            for session, runs in sessions.items():
                for run_name, raw_original in runs.items():
                    raw = raw_original.copy()
                    selected_channels = self._pick_channel_names(raw.ch_names)
                    raw.pick(selected_channels)
                    if self.srate is not None and float(raw.info["sfreq"]) != float(self.srate):
                        raw.resample(float(self.srate))
                    sfreq = float(raw.info["sfreq"])
                    samples_per_epoch = int(round(sfreq * self.epoch_sec))
                    for epoch_index, annotation in enumerate(raw.annotations):
                        try:
                            stage_5class = EVENT_ID_TO_STAGE[int(annotation["description"])]
                        except (KeyError, TypeError, ValueError):
                            continue
                        stage = map_stage_for_task(stage_5class, self.task)
                        if stage not in event_arrays:
                            continue
                        start = int(round(float(annotation["onset"]) * sfreq))
                        stop = start + samples_per_epoch
                        if start < 0 or stop > raw.n_times:
                            continue
                        epoch_uv = raw.get_data(start=start, stop=stop) * 1e6
                        event_arrays[stage].append(epoch_uv)
                        event_meta[stage].append(
                            {
                                "subject": subject,
                                "session": session,
                                "run": run_name,
                                "stage": stage,
                                "stage_5class": stage_5class,
                                "epoch_index": epoch_index,
                                "onset_sec": float(annotation["onset"]),
                                "dataset": dataset.dataset_code,
                            }
                        )

        if not any(event_arrays.values()):
            raise ValueError("no valid 30-second sleep staging events were found")
        classes = task_classes(self.task)
        xs: dict[str, np.ndarray] = {}
        ys: dict[str, np.ndarray] = {}
        metas: dict[str, pd.DataFrame] = {}
        class_index = {stage: index for index, stage in enumerate(classes)}
        for stage in classes:
            if not event_arrays[stage]:
                continue
            xs[stage] = np.stack(event_arrays[stage], axis=0)
            label_value = class_index[stage] if label_encode else stage
            ys[stage] = np.asarray([label_value] * len(event_arrays[stage]))
            metas[stage] = pd.DataFrame(event_meta[stage])
        if not return_concat:
            return xs, ys, metas
        return (
            np.concatenate([xs[stage] for stage in classes if stage in xs], axis=0),
            np.concatenate([ys[stage] for stage in classes if stage in ys], axis=0),
            pd.concat([metas[stage] for stage in classes if stage in metas], ignore_index=True),
        )

    def _pick_channel_names(self, available: list[str]) -> list[str]:
        if self.select_channels is None:
            return list(available)
        lookup = {_normalize_channel(channel): channel for channel in available}
        selected = []
        missing = []
        for requested in self.select_channels:
            normalized = _normalize_channel(requested)
            match = lookup.get(normalized)
            if match is None:
                candidates = [
                    actual
                    for key, actual in lookup.items()
                    if key.endswith(normalized) or normalized.endswith(key)
                ]
                match = candidates[0] if len(candidates) == 1 else None
            if match is None:
                missing.append(requested)
            else:
                selected.append(match)
        if missing:
            raise ValueError(f"requested channels not found: {missing}; available={available}")
        return selected


def _normalize_channel(name: str) -> str:
    return "".join(character.lower() for character in str(name) if character.isalnum())
