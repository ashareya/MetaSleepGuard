"""ISRUC-Sleep exposed through MetaBCI's ``BaseDataset`` contract."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

try:
    import mne
    from metabci.brainda.datasets.base import BaseDataset
except ImportError as exc:  # pragma: no cover
    raise ImportError("ISRUCSleep requires MetaBCI Brainda and MNE") from exc

from MetaSleepGuard.datasets.public_sleep.loaders import find_isruc_records, parse_isruc_labels
from .sleep_edf import STAGE_EVENT_IDS


class ISRUCSleep(BaseDataset):
    """Local ISRUC EDF recordings with adjacent expert-label files."""

    def __init__(
        self,
        root: str | Path,
        channels: list[str] | tuple[str, ...] | None = None,
        max_subjects: int | None = None,
        epoch_sec: float = 30.0,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise ValueError(f"ISRUC root does not exist: {self.root}")
        if float(epoch_sec) != 30.0:
            raise ValueError("ISRUCSleep currently supports 30-second windows only")
        self.epoch_sec = float(epoch_sec)
        self.requested_channels = list(channels) if channels is not None else None
        grouped = defaultdict(list)
        for edf, label in find_isruc_records(self.root):
            if label is not None:
                grouped[self._subject_id(edf)].append((edf, label))
        subjects = sorted(grouped)
        if max_subjects is not None:
            if max_subjects <= 0:
                raise ValueError("max_subjects must be positive")
            subjects = subjects[:max_subjects]
        if not subjects:
            raise ValueError(f"No ISRUC EDF/label pairs found under {self.root}")
        self._pairs = {subject: grouped[subject] for subject in subjects}
        probe = mne.io.read_raw_edf(self._pairs[subjects[0]][0][0], preload=False, verbose="ERROR")
        selected = self._select_channels(probe.ch_names)
        super().__init__(
            dataset_code="ISRUC-MetaSleepGuard",
            subjects=subjects,
            events={stage: (event_id, (0.0, 30.0)) for stage, event_id in STAGE_EVENT_IDS.items()},
            channels=selected,
            srate=float(probe.info["sfreq"]),
            paradigm="sleep_staging",
        )

    @staticmethod
    def _subject_id(path: Path) -> str:
        for part in reversed(path.parent.parts):
            if any(character.isdigit() for character in part):
                return part
        return path.stem

    def _select_channels(self, available) -> list[str]:
        available = list(available)
        if self.requested_channels is None:
            return available
        lookup = {name.upper(): name for name in available}
        missing = [name for name in self.requested_channels if name.upper() not in lookup]
        if missing:
            raise ValueError(f"ISRUC channels not found: {missing}; available={available}")
        return [lookup[name.upper()] for name in self.requested_channels]

    def data_path(
        self,
        subject: Union[str, int],
        path: Optional[Union[str, Path]] = None,
        force_update: bool = False,
        update_path: Optional[bool] = None,
        proxies: Optional[dict[str, str]] = None,
        verbose: Optional[Union[bool, str, int]] = None,
    ):
        del path, force_update, update_path, proxies, verbose
        if subject not in self.subjects:
            raise ValueError(f"Invalid subject {subject!r}")
        return [[edf, labels] for edf, labels in self._pairs[str(subject)]]

    def _get_single_subject_data(self, subject, verbose=None):
        del verbose
        if subject not in self.subjects:
            raise ValueError(f"Invalid subject {subject!r}")
        runs = {}
        for index, (edf, label_file) in enumerate(self._pairs[str(subject)]):
            raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
            raw.pick(self._select_channels(raw.ch_names))
            labels = parse_isruc_labels(label_file)
            max_epochs = min(len(labels), int(raw.n_times // round(raw.info["sfreq"] * self.epoch_sec)))
            onsets, descriptions = [], []
            for epoch_index, stage in enumerate(labels[:max_epochs]):
                if stage in STAGE_EVENT_IDS:
                    onsets.append(epoch_index * self.epoch_sec)
                    descriptions.append(str(STAGE_EVENT_IDS[stage]))
            raw.set_annotations(
                mne.Annotations(
                    onset=onsets,
                    duration=[self.epoch_sec] * len(onsets),
                    description=descriptions,
                    orig_time=raw.info.get("meas_date"),
                )
            )
            runs[f"run_{index + 1}"] = raw
        return {"session_1": runs}
