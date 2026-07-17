"""Public sleep dataset loaders.

The real loaders read Sleep-EDF and ISRUC-like files when data are present.
For tests and demos without protected data, :func:`generate_synthetic_public_records`
creates clearly marked synthetic records with the same in-memory shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import logging
import re
from typing import Iterable, Sequence

import numpy as np

from ...preprocessing.label_mapping import map_raw_stage

LOGGER = logging.getLogger(__name__)


@dataclass
class SleepRecord:
    """One subject-level EEG record and its 30-second expert labels."""

    subject_id: str
    dataset: str
    signals: np.ndarray
    sfreq: float
    channel_names: list[str]
    labels: list[str]
    epoch_sec: float = 30.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.signals = np.asarray(self.signals, dtype=float)
        if self.signals.ndim != 2:
            raise ValueError("signals must have shape (channels, samples)")
        if len(self.channel_names) != self.signals.shape[0]:
            raise ValueError("channel_names length must match signal channels")

    @property
    def n_epochs_from_signal(self) -> int:
        return int(self.signals.shape[1] // int(round(self.sfreq * self.epoch_sec)))


def find_sleep_edf_pairs(root: str | Path) -> list[tuple[Path, Path]]:
    """Return PSG/hypnogram EDF pairs under a Sleep-EDF directory."""

    root = Path(root)
    psg_files = sorted(root.rglob("*PSG.edf"))
    hypnograms = sorted(root.rglob("*Hypnogram.edf"))
    pairs: list[tuple[Path, Path]] = []
    for psg in psg_files:
        match = re.match(r"((?:SC|ST)\d{4})", psg.name.upper())
        prefix = match.group(1) if match else psg.stem.split("-")[0]
        candidates = [
            hyp
            for hyp in hypnograms
            if hyp.parent == psg.parent and hyp.name.upper().startswith(prefix)
        ]
        if candidates:
            pairs.append((psg, candidates[0]))
    return pairs


def load_sleep_edf(
    root: str | Path,
    channels: Sequence[str] | None = None,
    max_subjects: int | None = None,
) -> list[SleepRecord]:
    """Load Sleep-EDF PSG files and annotation hypnograms.

    Labels are normalized to ``W/N1/N2/N3/REM``; unscored and movement epochs are
    kept as ``UNKNOWN`` so downstream code can drop them explicitly.
    """

    try:
        import mne
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError("mne is required to read Sleep-EDF files") from exc

    records: list[SleepRecord] = []
    included_subjects: set[str] = set()
    for psg_path, hyp_path in find_sleep_edf_pairs(root):
        subject_id = _sleep_edf_subject_id(psg_path)
        if subject_id not in included_subjects and max_subjects is not None and len(included_subjects) >= max_subjects:
            continue
        included_subjects.add(subject_id)
        LOGGER.info("Loading Sleep-EDF PSG=%s hypnogram=%s", psg_path, hyp_path)
        raw = mne.io.read_raw_edf(psg_path, preload=True, verbose="ERROR")
        if channels:
            available = [ch for ch in channels if ch in raw.ch_names]
            if available:
                raw.pick_channels(available, ordered=True)
        annotations = mne.read_annotations(hyp_path)
        raw.set_annotations(annotations)
        labels = _annotations_to_epoch_labels(raw.annotations, raw.n_times, raw.info["sfreq"])
        signals = raw.get_data()
        records.append(
            SleepRecord(
                subject_id=subject_id,
                dataset="sleep-edf",
                signals=signals,
                sfreq=float(raw.info["sfreq"]),
                channel_names=list(raw.ch_names),
                labels=labels,
                metadata={"psg_path": str(psg_path), "hypnogram_path": str(hyp_path)},
            )
        )
    return records


def find_isruc_records(root: str | Path, scorer: int = 1) -> list[tuple[Path, Path | None]]:
    """Find official ISRUC ``.rec``/EDF files and the requested scorer labels."""

    root = Path(root)
    if scorer not in {1, 2}:
        raise ValueError("scorer must be 1 or 2")
    edfs = sorted({*root.rglob("*.edf"), *root.rglob("*.rec")})
    records: list[tuple[Path, Path | None]] = []
    for edf in edfs:
        label_candidates: list[Path] = []
        for suffix in ("*.txt", "*.csv", "*.tsv", "*.xls", "*.xlsx"):
            label_candidates.extend(edf.parent.glob(suffix))
        label = _best_label_candidate(edf, label_candidates, scorer=scorer)
        records.append((edf, label))
    return records


def load_isruc_sleep(
    root: str | Path,
    channels: Sequence[str] | None = None,
    max_subjects: int | None = None,
) -> list[SleepRecord]:
    """Load ISRUC-Sleep EDF files plus expert labels from text/CSV sidecars."""

    try:
        import mne
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError("mne is required to read ISRUC EDF files") from exc

    records: list[SleepRecord] = []
    included_subjects: set[str] = set()
    scorer2_lookup = {edf.resolve(): label for edf, label in find_isruc_records(root, scorer=2)}
    for edf_path, label_path in find_isruc_records(root, scorer=1):
        subject_id = _subject_from_path(edf_path)
        if subject_id not in included_subjects and max_subjects is not None and len(included_subjects) >= max_subjects:
            continue
        included_subjects.add(subject_id)
        LOGGER.info("Loading ISRUC EDF=%s labels=%s", edf_path, label_path)
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
        if channels:
            available = [ch for ch in channels if ch in raw.ch_names]
            if available:
                raw.pick_channels(available, ordered=True)
        labels = parse_isruc_labels(label_path) if label_path else []
        scorer2_path = scorer2_lookup.get(edf_path.resolve())
        scorer2_labels = parse_isruc_labels(scorer2_path) if scorer2_path else []
        agreement = _label_agreement(labels, scorer2_labels)
        records.append(
            SleepRecord(
                subject_id=subject_id,
                dataset="isruc-sleep",
                signals=raw.get_data(),
                sfreq=float(raw.info["sfreq"]),
                channel_names=list(raw.ch_names),
                labels=labels,
                metadata={
                    "edf_path": str(edf_path),
                    "label_path": str(label_path) if label_path else "",
                    "scorer": 1,
                    "scorer2_label_path": str(scorer2_path) if scorer2_path else "",
                    "scorer_agreement": agreement,
                    "edf_sha256": _sha256(edf_path),
                    "label_sha256": _sha256(label_path) if label_path else "",
                },
            )
        )
    return records


def parse_isruc_labels(path: str | Path) -> list[str]:
    """Parse common ISRUC label text formats into canonical stages."""

    path = Path(path)
    if not path.exists():
        return []
    if path.suffix.lower() in {".xls", ".xlsx"}:
        try:
            import pandas as pd
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("pandas with an Excel reader is required for ISRUC Excel labels") from exc
        table = pd.read_excel(path, header=None)
        return _labels_from_rows(table.astype(object).values.tolist())
    labels: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = re.split(r"[\s,;\t]+", line.strip())
            tokens = [p for p in parts if p != ""]
            if not tokens:
                continue
            stage = _stage_from_tokens(tokens)
            if stage is not None:
                labels.append(stage)
    return labels


def generate_synthetic_public_records(
    dataset: str = "sleep-edf",
    n_subjects: int = 4,
    n_epochs: int = 12,
    sfreq: float = 250.0,
    n_channels: int = 2,
    seed: int = 7,
) -> list[SleepRecord]:
    """Create labeled synthetic records for smoke tests and demos.

    The returned metadata contains ``synthetic=True``; these records must not be
    reported as public-data accuracy results.
    """

    rng = np.random.default_rng(seed)
    stages = ["W", "N1", "N2", "N3", "REM"]
    samples_per_epoch = int(round(sfreq * 30.0))
    records: list[SleepRecord] = []
    channel_names = [f"EEG{i + 1}" for i in range(n_channels)]
    freqs_by_stage = {"W": 10.0, "N1": 6.0, "N2": 13.5, "N3": 2.0, "REM": 7.0}
    for subject_index in range(n_subjects):
        labels = [stages[(subject_index + epoch_index) % len(stages)] for epoch_index in range(n_epochs)]
        signal = np.zeros((n_channels, n_epochs * samples_per_epoch), dtype=float)
        t = np.arange(samples_per_epoch) / sfreq
        for epoch_index, stage in enumerate(labels):
            base = freqs_by_stage[stage]
            for ch in range(n_channels):
                amp = 20e-6 + ch * 4e-6
                noise = rng.normal(scale=5e-6, size=samples_per_epoch)
                segment = amp * np.sin(2 * np.pi * (base + ch * 0.7) * t) + noise
                if stage == "N3":
                    segment += 35e-6 * np.sin(2 * np.pi * 1.2 * t)
                start = epoch_index * samples_per_epoch
                signal[ch, start : start + samples_per_epoch] = segment
        records.append(
            SleepRecord(
                subject_id=f"{dataset}_synthetic_{subject_index:02d}",
                dataset=dataset,
                signals=signal,
                sfreq=sfreq,
                channel_names=channel_names,
                labels=labels,
                metadata={"synthetic": True},
            )
        )
    return records


def _annotations_to_epoch_labels(annotations, n_times: int, sfreq: float, epoch_sec: float = 30.0) -> list[str]:
    n_epochs = int(n_times // int(round(sfreq * epoch_sec)))
    labels = ["UNKNOWN"] * n_epochs
    for onset, duration, description in zip(annotations.onset, annotations.duration, annotations.description):
        start = max(0, int(np.floor(onset / epoch_sec)))
        stop = min(n_epochs, int(np.ceil((onset + duration) / epoch_sec)))
        stage = map_raw_stage(str(description))
        for epoch_index in range(start, stop):
            labels[epoch_index] = stage
    return labels


def _sleep_edf_subject_id(path: Path) -> str:
    name = path.name.upper()
    match = re.match(r"((?:SC|ST)\d{3})\d", name)
    if match:
        return match.group(1)
    fallback = re.search(r"(?:SC|ST)(\d+)", name)
    return fallback.group(0) if fallback else path.stem


def _subject_from_path(path: Path) -> str:
    for part in reversed(path.parent.parts):
        if re.search(r"\d", part):
            return part
    match = re.match(r"(.+?)(?:[_-](?:night|recording)?\d+)?$", path.stem, flags=re.IGNORECASE)
    return match.group(1) if match else path.stem


def _best_label_candidate(edf: Path, candidates: Iterable[Path], scorer: int = 1) -> Path | None:
    candidates = list(candidates)
    if not candidates:
        return None
    stem = edf.stem.lower()
    def score(path: Path) -> tuple[int, int]:
        candidate = path.stem.lower()
        scorer_tokens = {f"{stem}_{scorer}", f"{stem}-{scorer}", f"{stem} scorer {scorer}"}
        if candidate in scorer_tokens:
            rank = 0
        elif re.search(rf"(?:^|[_\-\s]){scorer}(?:$|[_\-\s])", candidate) and stem in candidate:
            rank = 1
        elif candidate == stem:
            rank = 2
        elif candidate.startswith(f"{stem}_") or candidate.startswith(f"{stem}-"):
            rank = 3
        elif stem in candidate:
            rank = 4
        else:
            rank = 10
        other = 2 if scorer == 1 else 1
        if re.search(rf"(?:^|[_\-\s]){other}(?:$|[_\-\s])", candidate):
            rank += 8
        if any(token in candidate for token in ("readme", "info", "description")):
            rank += 20
        return rank, len(path.name)

    scored = sorted(candidates, key=score)
    return scored[0] if score(scored[0])[0] < 10 else None


def _stage_from_tokens(tokens: Sequence[str]) -> str | None:
    for token in reversed(tokens):
        cleaned = str(token).strip()
        if not cleaned or cleaned.lower() in {"stage", "label", "epoch", "nan"}:
            continue
        stage = map_raw_stage(cleaned)
        if stage != "UNKNOWN" or cleaned.upper() in {"UNKNOWN", "U", "?", "-1", "6", "9"}:
            return stage
    return None


def _labels_from_rows(rows: Iterable[Sequence[object]]) -> list[str]:
    labels: list[str] = []
    for row in rows:
        stage = _stage_from_tokens([str(value) for value in row if value is not None])
        if stage is not None:
            labels.append(stage)
    return labels


def _label_agreement(first: Sequence[str], second: Sequence[str]) -> dict:
    length = min(len(first), len(second))
    if length == 0:
        return {"n_compared": 0, "agreement": None}
    a = np.asarray(first[:length])
    b = np.asarray(second[:length])
    valid = (a != "UNKNOWN") & (b != "UNKNOWN")
    return {
        "n_compared": int(np.sum(valid)),
        "agreement": float(np.mean(a[valid] == b[valid])) if np.any(valid) else None,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
