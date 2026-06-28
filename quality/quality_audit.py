"""Window-level quality audit and BDF/FIF file auditing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import hashlib
import json
import logging
from typing import Sequence

import numpy as np

from ..preprocessing.epoching import epoch_signal
from .artifact_rules import (
    detect_abnormal_amplitude,
    detect_bad_channel,
    detect_bad_channel_names,
    detect_baseline_drift,
    detect_data_dropout,
    detect_flatline,
    detect_line_noise,
    detect_motion_artifact,
    detect_saturation,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class QualityResult:
    window_start_time: float
    window_end_time: float
    channel_names: list[str]
    quality_score: float
    quality_grade: str
    bad_flags: list[str]
    is_reliable: bool
    reason: str

    def to_row(self) -> dict:
        row = asdict(self)
        row["channel_names"] = "|".join(self.channel_names)
        row["bad_flags"] = "|".join(self.bad_flags)
        return row


def audit_epoch(
    epoch: np.ndarray,
    sfreq: float,
    channel_names: Sequence[str] | None = None,
    start_time: float = 0.0,
    epoch_sec: float = 30.0,
) -> QualityResult:
    """Audit one epoch and return score, grade, flags, and reason."""

    epoch = np.asarray(epoch, dtype=float)
    names = list(channel_names or [f"ch{i + 1}" for i in range(epoch.shape[0])])
    detectors = {
        "line_noise": detect_line_noise(epoch, sfreq),
        "baseline_drift": detect_baseline_drift(epoch, sfreq),
        "channel_saturation": detect_saturation(epoch),
        "flatline": detect_flatline(epoch),
        "motion_artifact": detect_motion_artifact(epoch),
        "data_dropout": detect_data_dropout(epoch),
        "abnormal_amplitude": detect_abnormal_amplitude(epoch),
        "bad_channel": detect_bad_channel(epoch),
    }
    bad_flags = [name for name, flagged in detectors.items() if flagged]
    penalty = sum(_FLAG_PENALTIES.get(flag, 12) for flag in bad_flags)
    score = float(max(0, 100 - penalty))
    grade = _grade_from_score(score)
    severe_flags = {
        "data_dropout",
        "flatline",
        "channel_saturation",
        "motion_artifact",
        "abnormal_amplitude",
        "bad_channel",
    }
    reliable = grade in {"A", "B", "C"} and not any(flag in bad_flags for flag in severe_flags)
    reason = "ok" if not bad_flags else ", ".join(bad_flags)
    return QualityResult(
        window_start_time=float(start_time),
        window_end_time=float(start_time + epoch_sec),
        channel_names=names,
        quality_score=score,
        quality_grade=grade,
        bad_flags=bad_flags,
        is_reliable=bool(reliable),
        reason=reason,
    )


def audit_windows(
    signals: np.ndarray,
    sfreq: float,
    channel_names: Sequence[str] | None = None,
    epoch_sec: float = 30.0,
) -> list[QualityResult]:
    epochs = epoch_signal(signals, sfreq, epoch_sec=epoch_sec)
    return [
        audit_epoch(epoch, sfreq, channel_names, start_time=i * epoch_sec, epoch_sec=epoch_sec)
        for i, epoch in enumerate(epochs)
    ]


def write_quality_csv(results: Sequence[QualityResult], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "window_start_time",
        "window_end_time",
        "channel_names",
        "quality_score",
        "quality_grade",
        "bad_flags",
        "is_reliable",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_row())
    return path


def audit_bdf_fif_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    max_windows_per_file: int = 5,
) -> Path:
    """Audit BDF/FIF files, generate CSV plus representative PNG figures."""

    try:
        import mne
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("mne is required for BDF/FIF audit") from exc

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_paths = sorted(list(input_dir.rglob("*.bdf")) + list(input_dir.rglob("*.fif")))
    hashes: dict[str, str] = {}
    rows: list[dict] = []
    for path in file_paths:
        sha1 = _sha1(path)
        duplicate_of = hashes.get(sha1, "")
        hashes.setdefault(sha1, str(path))
        raw = _read_mne_raw(mne, path)
        sfreq = float(raw.info["sfreq"])
        duration = float(raw.n_times / sfreq)
        audit_stop = min(raw.n_times, int(round(max_windows_per_file * 30.0 * sfreq)))
        data = raw.get_data(start=0, stop=audit_stop)
        quality = audit_windows(data, sfreq, raw.ch_names)[:max_windows_per_file]
        flags = sorted({flag for result in quality for flag in result.bad_flags})
        detected_bad_channels = detect_bad_channel_names(data, raw.ch_names)
        bad_channel_names = sorted(set(raw.info.get("bads", [])) | set(detected_bad_channels))
        worst_grade = _worst_grade(result.quality_grade for result in quality)
        representative_plot = output_dir / f"{path.stem}_{sha1[:8]}_representative.png"
        rows.append(
            {
                "file_path": str(path),
                "sha1": sha1,
                "duplicate_of": duplicate_of,
                "n_channels": len(raw.ch_names),
                "channel_names": "|".join(raw.ch_names),
                "sfreq": sfreq,
                "duration_sec": duration,
                "bad_channel_names": "|".join(bad_channel_names),
                "quality_flags": "|".join(flags),
                "min_quality_grade": worst_grade,
                "worst_quality_grade": worst_grade,
                "representative_plot": str(representative_plot),
            }
        )
        _plot_representative(raw, representative_plot)
    csv_path = output_dir / "bdf_fif_audit.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("file_path,sha1,duplicate_of,n_channels,channel_names,sfreq,duration_sec,bad_channel_names,quality_flags,min_quality_grade,worst_quality_grade,representative_plot\n", encoding="utf-8")
    LOGGER.info("BDF/FIF audit written to %s", csv_path)
    return csv_path


def quality_results_to_json(results: Sequence[QualityResult]) -> str:
    return json.dumps([result.to_row() for result in results], ensure_ascii=False, indent=2)


_FLAG_PENALTIES = {
    "data_dropout": 45,
    "flatline": 45,
    "channel_saturation": 30,
    "abnormal_amplitude": 25,
    "motion_artifact": 20,
    "baseline_drift": 15,
    "line_noise": 15,
    "bad_channel": 20,
}


def _grade_from_score(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 50:
        return "C"
    return "D"


def _worst_grade(grades) -> str:
    rank = {"A": 0, "B": 1, "C": 2, "D": 3}
    values = list(grades)
    return max(values, key=lambda grade: rank.get(grade, -1)) if values else "NA"


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_mne_raw(mne, path: Path):
    suffix = path.suffix.lower()
    if suffix == ".bdf":
        return mne.io.read_raw_bdf(path, preload=False, verbose="ERROR")
    if suffix == ".fif":
        return mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
    raise ValueError(f"unsupported audit file type: {suffix}")


def _plot_representative(raw, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    sfreq = float(raw.info["sfreq"])
    spectrum_samples = min(raw.n_times, int(sfreq * 30))
    data = raw.get_data(picks=list(range(min(2, len(raw.ch_names)))), start=0, stop=spectrum_samples)
    samples = min(data.shape[1], int(sfreq * 10))
    time = np.arange(samples) / sfreq
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    for i, channel in enumerate(data):
        axes[0].plot(time, channel[:samples], label=raw.ch_names[i], linewidth=0.8)
        freqs, psd = welch(channel, fs=sfreq, nperseg=min(channel.size, int(sfreq * 4)))
        axes[1].semilogy(freqs, psd, label=raw.ch_names[i], linewidth=0.8)
    axes[0].set_title("Representative waveform")
    axes[0].set_xlabel("Time (s)")
    axes[1].set_title("Representative spectrum")
    axes[1].set_xlim(0, min(60, sfreq / 2))
    axes[1].set_xlabel("Frequency (Hz)")
    for ax in axes:
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)
