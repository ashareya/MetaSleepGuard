"""Sleep-EDF dataset exposed through MetaBCI's ``BaseDataset`` contract."""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from typing import Optional, Union

try:
    import mne
    from metabci.brainda.datasets.base import BaseDataset
except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
    raise ImportError(
        "SleepEDF requires MetaBCI Brainda and MNE. Install metabci-sleep with its dependencies."
    ) from exc

from MetaSleepGuard.datasets.public_sleep.loaders import find_sleep_edf_pairs
from MetaSleepGuard.preprocessing.label_mapping import map_raw_stage


STAGE_EVENT_IDS = {"W": 1, "N1": 2, "N2": 3, "N3": 4, "REM": 5}


def _subject_id(path: Path) -> str:
    name = path.name.upper()
    if name.startswith(("SC", "ST")) and len(name) >= 5:
        return name[:5]
    return path.stem.split("-")[0]


class SleepEDF(BaseDataset):
    """Local Sleep-EDF PSG and hypnogram pairs as a MetaBCI dataset.

    Parameters
    ----------
    root:
        Directory containing ``*PSG.edf`` and matching ``*Hypnogram.edf`` files.
    channels:
        Ordered EEG channels to expose. If ``None``, all PSG channels are kept.
    max_subjects:
        Optional limit applied after deterministic subject discovery.
    epoch_sec:
        Sleep staging window length. Version 0.1 supports the standard 30 seconds.
    """

    def __init__(
        self,
        root: str | Path,
        channels: list[str] | tuple[str, ...] | None = None,
        max_subjects: int | None = None,
        epoch_sec: float = 30.0,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise ValueError(f"Sleep-EDF root does not exist: {self.root}")
        if float(epoch_sec) != 30.0:
            raise ValueError("SleepEDF currently supports 30-second staging windows only")
        self.epoch_sec = float(epoch_sec)
        self.requested_channels = list(channels) if channels is not None else None

        grouped: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
        for psg_path, hypnogram_path in find_sleep_edf_pairs(self.root):
            grouped[_subject_id(psg_path)].append((psg_path, hypnogram_path))
        subjects = sorted(grouped)
        if max_subjects is not None:
            if max_subjects <= 0:
                raise ValueError("max_subjects must be positive")
            subjects = subjects[: int(max_subjects)]
        if not subjects:
            raise ValueError(f"No Sleep-EDF PSG/hypnogram pairs found under {self.root}")
        self._pairs_by_subject = {subject: sorted(grouped[subject]) for subject in subjects}

        probe = mne.io.read_raw_edf(self._pairs_by_subject[subjects[0]][0][0], preload=False, verbose="ERROR")
        available_channels = list(probe.ch_names)
        selected_channels = self._validate_channels(available_channels)
        super().__init__(
            dataset_code="SleepEDF-MetaSleepGuard",
            subjects=subjects,
            events={stage: (event_id, (0.0, self.epoch_sec)) for stage, event_id in STAGE_EVENT_IDS.items()},
            channels=selected_channels,
            srate=float(probe.info["sfreq"]),
            paradigm="sleep_staging",
        )

    def _validate_channels(self, available: list[str]) -> list[str]:
        if self.requested_channels is None:
            return available
        lookup = {_normalize_channel(name): name for name in available}
        selected = []
        missing = []
        for requested in self.requested_channels:
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
            raise ValueError(f"Sleep-EDF channels not found: {missing}; available={available}")
        return selected

    def data_path(
        self,
        subject: Union[str, int],
        path: Optional[Union[str, Path]] = None,
        force_update: bool = False,
        update_path: Optional[bool] = None,
        proxies: Optional[dict[str, str]] = None,
        verbose: Optional[Union[bool, str, int]] = None,
    ) -> list[list[Union[str, Path]]]:
        """Return local ``[PSG, hypnogram]`` pairs, one list per recording."""

        del path, force_update, update_path, proxies, verbose
        if subject not in self.subjects:
            raise ValueError(f"Invalid subject {subject!r}; valid subjects={self.subjects}")
        return [[psg, hypnogram] for psg, hypnogram in self._pairs_by_subject[str(subject)]]

    def _get_single_subject_data(
        self,
        subject: Union[str, int],
        verbose: Optional[Union[bool, str, int]] = None,
    ) -> dict[str, dict[str, mne.io.BaseRaw]]:
        del verbose
        if subject not in self.subjects:
            raise ValueError(f"Invalid subject {subject!r}; valid subjects={self.subjects}")
        runs: dict[str, mne.io.BaseRaw] = {}
        for index, (psg_path, hypnogram_path) in enumerate(self._pairs_by_subject[str(subject)]):
            raw = mne.io.read_raw_edf(psg_path, preload=True, verbose="ERROR")
            selected = self._validate_channels(list(raw.ch_names))
            raw.pick(selected)
            hypnogram = mne.read_annotations(hypnogram_path)
            raw.set_annotations(self._expanded_stage_annotations(raw, hypnogram))
            raw.info["description"] = (
                f"SleepEDF subject={subject}; psg={psg_path}; hypnogram={hypnogram_path}; "
                "annotations are 30-second numeric stage events"
            )
            runs[f"run_{index + 1}"] = raw
        return {"session_1": runs}

    def _expanded_stage_annotations(self, raw, annotations) -> mne.Annotations:
        n_epochs = int(raw.n_times // int(round(float(raw.info["sfreq"]) * self.epoch_sec)))
        labels = ["UNKNOWN"] * n_epochs
        for onset, duration, description in zip(
            annotations.onset, annotations.duration, annotations.description
        ):
            start = max(0, int(onset // self.epoch_sec))
            stop = min(n_epochs, int(math.ceil((onset + duration) / self.epoch_sec)))
            stage = map_raw_stage(description)
            for epoch_index in range(start, stop):
                labels[epoch_index] = stage
        onsets = []
        descriptions = []
        for epoch_index, stage in enumerate(labels):
            event_id = STAGE_EVENT_IDS.get(stage)
            if event_id is None:
                continue
            onsets.append(epoch_index * self.epoch_sec)
            descriptions.append(str(event_id))
        return mne.Annotations(
            onset=onsets,
            duration=[self.epoch_sec] * len(onsets),
            description=descriptions,
            orig_time=raw.info.get("meas_date"),
        )


def _normalize_channel(name: str) -> str:
    return "".join(character.lower() for character in str(name) if character.isalnum())
