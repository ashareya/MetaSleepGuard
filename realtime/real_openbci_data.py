"""Real OpenBCI GUI TXT parsing, temporal coverage, and quality auditing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch


ACTIVE_EXG_COLUMNS = ("EXG Channel 1", "EXG Channel 6")
ACTIVE_GUI_CHANNELS = ("Ch2 / Fp1", "Ch7 / Fp2")
REAL_OPENBCI_QUALITY_RULES = {
    "reject_if_temporal_or_sample_ratio_below": 0.8,
    "mark_data_dropout_if_temporal_coverage_below": 0.995,
    "mark_sample_loss_if_sample_completeness_below": 0.9,
    "line_noise_ratio_threshold": 0.2,
    "strong_line_noise_ratio_threshold": 0.5,
    "baseline_drift_ratio_threshold": 0.5,
    "motion_filtered_abs99_uv_threshold": 500.0,
    "motion_filtered_diff99_uv_threshold": 50.0,
    "abnormal_filtered_abs99_uv_threshold": 1000.0,
    "saturation_absolute_uv_threshold": 180000.0,
    "protocol_artifact_stages_forced_to_grade_c": ["blink", "jaw", "head_turn", "cable_move"],
}


@dataclass(frozen=True)
class OpenBCIHeader:
    number_of_channels: int
    sample_rate: float
    board: str
    header_lines: list[str]


@dataclass
class OpenBCISegment:
    path: Path
    header: OpenBCIHeader
    data: pd.DataFrame
    active_exg_channels: list[str]
    inactive_exg_channels: list[str]
    observed_nonzero_exg_channels: list[str]
    unexpected_nonzero_inactive_channels: list[str]
    start_timestamp: float
    end_timestamp: float

    @property
    def row_count(self) -> int:
        return int(len(self.data))

    @property
    def duration_sec(self) -> float:
        return float(max(0.0, self.end_timestamp - self.start_timestamp))

    def summary(self) -> dict:
        return {
            "file_name": self.path.name,
            "source_path": str(self.path),
            "file_size_bytes": self.path.stat().st_size,
            "sample_rate_hz": self.header.sample_rate,
            "declared_exg_channels": self.header.number_of_channels,
            "board": self.header.board,
            "row_count": self.row_count,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "start_time": format_timestamp(self.start_timestamp),
            "end_time": format_timestamp(self.end_timestamp),
            "duration_sec": self.duration_sec,
            "active_exg_channels": "|".join(self.active_exg_channels),
            "inactive_exg_channels": "|".join(self.inactive_exg_channels),
            "observed_nonzero_exg_channels": "|".join(self.observed_nonzero_exg_channels),
            "unexpected_nonzero_inactive_channels": "|".join(self.unexpected_nonzero_inactive_channels),
        }


def parse_openbci_header(path: str | Path) -> OpenBCIHeader:
    path = Path(path)
    header_lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("%"):
                break
            header_lines.append(line.rstrip("\r\n"))
    text = "\n".join(header_lines)
    channels = _extract_number(text, r"Number of channels\s*=\s*(\d+)", int, "number of channels")
    sample_rate = _extract_number(text, r"Sample Rate\s*=\s*([0-9.]+)", float, "sample rate")
    board_match = re.search(r"Board\s*=\s*(.+)", text)
    board = board_match.group(1).strip() if board_match else "unknown"
    return OpenBCIHeader(int(channels), float(sample_rate), board, header_lines)


def read_openbci_segment(
    path: str | Path,
    selected_columns: Sequence[str] = ACTIVE_EXG_COLUMNS,
    chunksize: int = 200_000,
) -> OpenBCISegment:
    """Read selected real channels while classifying all EXG channels as active/inactive."""

    path = Path(path)
    header = parse_openbci_header(path)
    exg_columns = [f"EXG Channel {index}" for index in range(header.number_of_channels)]
    required = [*exg_columns, "Timestamp"]
    frames: list[pd.DataFrame] = []
    sums = np.zeros(len(exg_columns), dtype=float)
    sums2 = np.zeros(len(exg_columns), dtype=float)
    finite_counts = np.zeros(len(exg_columns), dtype=np.int64)
    nonzero_counts = np.zeros(len(exg_columns), dtype=np.int64)
    for chunk in pd.read_csv(
        path,
        comment="%",
        usecols=required,
        chunksize=chunksize,
        skipinitialspace=True,
    ):
        values = chunk[exg_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        finite = np.isfinite(values)
        safe = np.where(finite, values, 0.0)
        sums += safe.sum(axis=0)
        sums2 += (safe * safe).sum(axis=0)
        finite_counts += finite.sum(axis=0)
        nonzero_counts += ((np.abs(safe) > 1e-12) & finite).sum(axis=0)
        frame = chunk[["Timestamp", *selected_columns]].copy()
        frame.columns = ["timestamp", "ch2_uv", "ch7_uv"]
        frames.append(frame)
    if not frames:
        raise ValueError(f"OpenBCI file contains no data rows: {path}")
    data = pd.concat(frames, ignore_index=True)
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data[np.isfinite(data["timestamp"])].sort_values("timestamp", kind="stable").reset_index(drop=True)
    if data.empty:
        raise ValueError(f"OpenBCI file contains no finite timestamps: {path}")
    counts = np.maximum(finite_counts, 1)
    means = sums / counts
    variances = np.maximum(0.0, sums2 / counts - means * means)
    stds = np.sqrt(variances)
    nonzero_ratio = nonzero_counts / counts
    observed_nonzero = [
        name for name, std, ratio in zip(exg_columns, stds, nonzero_ratio) if std > 0.01 and ratio > 0.001
    ]
    active = [name for name in selected_columns if name in observed_nonzero]
    inactive = [name for name in exg_columns if name not in selected_columns]
    unexpected_nonzero = [name for name in observed_nonzero if name in inactive]
    sfreq = header.sample_rate
    return OpenBCISegment(
        path=path,
        header=header,
        data=data,
        active_exg_channels=active,
        inactive_exg_channels=inactive,
        observed_nonzero_exg_channels=observed_nonzero,
        unexpected_nonzero_inactive_channels=unexpected_nonzero,
        start_timestamp=float(data["timestamp"].iloc[0]),
        end_timestamp=float(data["timestamp"].iloc[-1] + 1.0 / sfreq),
    )


def combine_segment_data(segments: Sequence[OpenBCISegment]) -> pd.DataFrame:
    frames = []
    for segment in segments:
        frame = segment.data.copy()
        frame["source_file"] = segment.path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["timestamp", "ch2_uv", "ch7_uv", "source_file"])
    return pd.concat(frames, ignore_index=True).sort_values("timestamp", kind="stable").reset_index(drop=True)


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((float(start), float(end)) for start, end in intervals if end > start)
    merged: list[list[float]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def clip_intervals(
    intervals: Iterable[tuple[float, float]],
    start: float,
    end: float,
) -> list[tuple[float, float]]:
    return merge_intervals((max(start, left), min(end, right)) for left, right in intervals if right > start and left < end)


def interval_coverage(intervals: Iterable[tuple[float, float]], start: float, end: float) -> float:
    return float(sum(right - left for left, right in clip_intervals(intervals, start, end)))


def find_gaps(intervals: Iterable[tuple[float, float]], start: float, end: float) -> list[tuple[float, float]]:
    merged = clip_intervals(intervals, start, end)
    gaps: list[tuple[float, float]] = []
    cursor = start
    for left, right in merged:
        if left > cursor:
            gaps.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < end:
        gaps.append((cursor, end))
    return gaps


def load_experiment_config(directory: str | Path) -> dict:
    candidates = sorted(Path(directory).glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"no experiment JSON config found in {directory}")
    return json.loads(candidates[0].read_text(encoding="utf-8"))


def load_event_log(directory: str | Path) -> pd.DataFrame:
    candidates = [path for path in Path(directory).glob("*.csv") if "30" not in path.name]
    if not candidates:
        raise FileNotFoundError(f"no event CSV found in {directory}")
    return pd.read_csv(candidates[0])


def load_window_log(directory: str | Path) -> pd.DataFrame:
    candidates = [path for path in Path(directory).glob("*.csv") if "30" in path.name]
    if not candidates:
        raise FileNotFoundError(f"no 30-second window CSV found in {directory}")
    return pd.read_csv(candidates[0])


def experiment_start_timestamp(events: pd.DataFrame) -> float:
    time_column = events.columns[0]
    return float(pd.Timestamp(events.iloc[0][time_column]).timestamp())


def stages_from_config(config: dict, experiment_start: float) -> list[dict]:
    stage_list = next(
        value
        for value in config.values()
        if isinstance(value, list) and value and isinstance(value[0], dict) and "start_sec" in value[0]
    )
    return [
        {
            **stage,
            "absolute_start": experiment_start + float(stage["start_sec"]),
            "absolute_end": experiment_start + float(stage["end_sec"]),
        }
        for stage in stage_list
    ]


def audit_real_openbci_window(
    signal_uv: np.ndarray,
    sfreq: float,
    temporal_coverage_ratio: float,
    sample_completeness_ratio: float,
    protocol_artifact: bool = False,
) -> dict:
    """Audit the two active OpenBCI channels without scoring disabled channels."""

    x = np.asarray(signal_uv, dtype=float)
    flags: list[str] = []
    metrics = {
        "ch2_std_uv": 0.0,
        "ch7_std_uv": 0.0,
        "ch2_ptp_uv": 0.0,
        "ch7_ptp_uv": 0.0,
        "ch2_filtered_abs99_uv": 0.0,
        "ch7_filtered_abs99_uv": 0.0,
        "ch2_line_noise_ratio": 0.0,
        "ch7_line_noise_ratio": 0.0,
        "ch2_drift_ratio": 0.0,
        "ch7_drift_ratio": 0.0,
    }
    rules = REAL_OPENBCI_QUALITY_RULES
    if temporal_coverage_ratio < rules["mark_data_dropout_if_temporal_coverage_below"]:
        flags.append("data_dropout")
    if (
        temporal_coverage_ratio < rules["reject_if_temporal_or_sample_ratio_below"]
        or sample_completeness_ratio < rules["reject_if_temporal_or_sample_ratio_below"]
    ):
        flags.append("effective_data_ratio_insufficient")
    if x.ndim != 2 or x.shape[0] != 2 or x.shape[1] < 8:
        flags.extend(["data_dropout", "zero_variance_channel"])
        return _quality_result(flags, metrics, temporal_coverage_ratio, sample_completeness_ratio, protocol_artifact)
    finite_columns = np.isfinite(x).all(axis=0)
    x = x[:, finite_columns]
    if x.shape[1] < 8:
        flags.extend(["data_dropout", "zero_variance_channel"])
        return _quality_result(flags, metrics, temporal_coverage_ratio, sample_completeness_ratio, protocol_artifact)
    centered = x - np.median(x, axis=1, keepdims=True)
    stds = np.std(centered, axis=1)
    ptp = np.ptp(centered, axis=1)
    metrics.update(
        {
            "ch2_std_uv": float(stds[0]),
            "ch7_std_uv": float(stds[1]),
            "ch2_ptp_uv": float(ptp[0]),
            "ch7_ptp_uv": float(ptp[1]),
        }
    )
    if np.any(stds < 0.05):
        flags.extend(["flatline", "zero_variance_channel"])
    if np.any(np.mean(np.abs(np.diff(centered, axis=1)) < 0.01, axis=1) > 0.98):
        flags.append("flatline")
    if np.any(np.abs(x) >= rules["saturation_absolute_uv_threshold"]) or np.any(_repeated_extreme_ratio(x) > 0.02):
        flags.append("channel_saturation")
    filtered = _bandpass(centered, sfreq)
    filtered_abs99 = np.percentile(np.abs(filtered), 99, axis=1)
    filtered_diff99 = np.percentile(np.abs(np.diff(filtered, axis=1)), 99, axis=1)
    metrics["ch2_filtered_abs99_uv"] = float(filtered_abs99[0])
    metrics["ch7_filtered_abs99_uv"] = float(filtered_abs99[1])
    line_ratios = []
    drift_ratios = []
    for channel in centered:
        line, drift = _spectral_ratios(channel, sfreq)
        line_ratios.append(line)
        drift_ratios.append(drift)
    metrics["ch2_line_noise_ratio"], metrics["ch7_line_noise_ratio"] = map(float, line_ratios)
    metrics["ch2_drift_ratio"], metrics["ch7_drift_ratio"] = map(float, drift_ratios)
    if max(line_ratios) > rules["strong_line_noise_ratio_threshold"]:
        flags.append("strong_line_noise")
    elif max(line_ratios) > rules["line_noise_ratio_threshold"]:
        flags.append("line_noise")
    if max(drift_ratios) > rules["baseline_drift_ratio_threshold"]:
        flags.append("baseline_drift")
    if np.any(filtered_abs99 > rules["abnormal_filtered_abs99_uv_threshold"]) or np.any(np.ptp(filtered, axis=1) > 5_000.0):
        flags.append("abnormal_amplitude")
    if (
        np.any(filtered_abs99 > rules["motion_filtered_abs99_uv_threshold"])
        or np.any(filtered_diff99 > rules["motion_filtered_diff99_uv_threshold"])
    ):
        flags.append("motion_artifact")
    if sample_completeness_ratio < rules["mark_sample_loss_if_sample_completeness_below"]:
        flags.append("sample_loss")
    return _quality_result(flags, metrics, temporal_coverage_ratio, sample_completeness_ratio, protocol_artifact)


def window_integrity_table(
    data: pd.DataFrame,
    segments: Sequence[OpenBCISegment],
    experiment_start: float,
    window_log: pd.DataFrame,
) -> pd.DataFrame:
    intervals = [(segment.start_timestamp, segment.end_timestamp) for segment in segments]
    sfreq = float(segments[0].header.sample_rate)
    rows: list[dict] = []
    for _, logged in window_log.iterrows():
        index = int(logged.iloc[0])
        relative_start = float(logged.iloc[1])
        relative_end = float(logged.iloc[2])
        start = experiment_start + relative_start
        end = experiment_start + relative_end
        duration = end - start
        temporal_coverage = interval_coverage(intervals, start, end)
        selected = data[(data["timestamp"] >= start) & (data["timestamp"] < end)]
        sample_ratio = min(1.0, len(selected) / max(1.0, duration * sfreq))
        quality = audit_real_openbci_window(
            selected[["ch2_uv", "ch7_uv"]].to_numpy(float).T,
            sfreq,
            temporal_coverage / duration,
            sample_ratio,
        )
        rows.append(
            {
                "window_index": index,
                "window_start_sec": relative_start,
                "window_end_sec": relative_end,
                "stage": str(logged.iloc[3]),
                "absolute_start": format_timestamp(start),
                "absolute_end": format_timestamp(end),
                "sample_count": int(len(selected)),
                "expected_samples": int(round(duration * sfreq)),
                "temporal_coverage_sec": temporal_coverage,
                "coverage_ratio": temporal_coverage / duration,
                "sample_completeness_ratio": sample_ratio,
                "effective_data_ratio": min(temporal_coverage / duration, sample_ratio),
                **quality,
            }
        )
    return pd.DataFrame(rows)


def stage_quality_table(
    data: pd.DataFrame,
    segments: Sequence[OpenBCISegment],
    stages: Sequence[dict],
) -> pd.DataFrame:
    intervals = [(segment.start_timestamp, segment.end_timestamp) for segment in segments]
    sfreq = float(segments[0].header.sample_rate)
    artifact_keys = set(REAL_OPENBCI_QUALITY_RULES["protocol_artifact_stages_forced_to_grade_c"])
    rows: list[dict] = []
    for stage in stages:
        start = float(stage["absolute_start"])
        end = float(stage["absolute_end"])
        duration = end - start
        selected = data[(data["timestamp"] >= start) & (data["timestamp"] < end)]
        temporal_coverage = interval_coverage(intervals, start, end)
        sample_ratio = min(1.0, len(selected) / max(1.0, duration * sfreq))
        protocol_artifact = stage.get("key") in artifact_keys
        quality = audit_real_openbci_window(
            selected[["ch2_uv", "ch7_uv"]].to_numpy(float).T,
            sfreq,
            temporal_coverage / duration,
            sample_ratio,
            protocol_artifact=protocol_artifact,
        )
        rows.append(
            {
                "stage_key": stage.get("key"),
                "stage_name": stage.get("name"),
                "start_sec": stage.get("start_sec"),
                "end_sec": stage.get("end_sec"),
                "absolute_start": format_timestamp(start),
                "absolute_end": format_timestamp(end),
                "sample_count": int(len(selected)),
                "expected_samples": int(round(duration * sfreq)),
                "temporal_coverage_sec": temporal_coverage,
                "coverage_ratio": temporal_coverage / duration,
                "sample_completeness_ratio": sample_ratio,
                "effective_data_ratio": min(temporal_coverage / duration, sample_ratio),
                "protocol_artifact_stage": protocol_artifact,
                **quality,
            }
        )
    return pd.DataFrame(rows)


def format_timestamp(timestamp: float) -> str:
    return pd.to_datetime(timestamp, unit="s", utc=True).tz_convert("Asia/Shanghai").isoformat(timespec="milliseconds")


def _extract_number(text: str, pattern: str, converter, label: str):
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"OpenBCI header is missing {label}")
    return converter(match.group(1))


def _bandpass(centered_uv: np.ndarray, sfreq: float) -> np.ndarray:
    if centered_uv.shape[1] < 32:
        return centered_uv
    sos = butter(4, [0.3 / (sfreq / 2.0), min(35.0, sfreq * 0.45) / (sfreq / 2.0)], btype="bandpass", output="sos")
    try:
        return sosfiltfilt(sos, centered_uv, axis=1)
    except ValueError:
        return centered_uv


def _spectral_ratios(channel_uv: np.ndarray, sfreq: float) -> tuple[float, float]:
    nperseg = min(channel_uv.size, int(round(sfreq * 30.0)))
    freqs, psd = welch(channel_uv, fs=sfreq, nperseg=max(16, nperseg))
    line = _band_power(freqs, psd, 49.0, 51.0)
    eeg = _band_power(freqs, psd, 0.5, min(45.0, sfreq / 2.0 - 1e-6)) + 1e-20
    drift = _band_power(freqs, psd, 0.03, 0.5)
    drift_reference = _band_power(freqs, psd, 0.5, min(35.0, sfreq / 2.0 - 1e-6)) + 1e-20
    return float(line / eeg), float(drift / drift_reference)


def _band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if np.sum(mask) < 2:
        return float(np.sum(psd[mask]))
    return float(np.trapezoid(psd[mask], freqs[mask]))


def _repeated_extreme_ratio(x: np.ndarray) -> np.ndarray:
    ratios = []
    for channel in x:
        rounded = np.round(channel, decimals=3)
        minimum = np.min(rounded)
        maximum = np.max(rounded)
        ratios.append(max(np.mean(rounded == minimum), np.mean(rounded == maximum)))
    return np.asarray(ratios)


def _quality_result(
    flags: Sequence[str],
    metrics: dict,
    temporal_coverage_ratio: float,
    sample_completeness_ratio: float,
    protocol_artifact: bool,
) -> dict:
    unique_flags = sorted(set(flags))
    if protocol_artifact:
        unique_flags.append("protocol_induced_artifact")
    severe_d = {"effective_data_ratio_insufficient", "flatline", "zero_variance_channel", "channel_saturation"}
    severe_c = {"strong_line_noise", "motion_artifact", "abnormal_amplitude", "protocol_induced_artifact"}
    if severe_d.intersection(unique_flags):
        grade = "D"
    elif severe_c.intersection(unique_flags):
        grade = "C"
    elif unique_flags or temporal_coverage_ratio < 0.98 or sample_completeness_ratio < 0.9:
        grade = "B"
    else:
        grade = "A"
    reject_ratio = REAL_OPENBCI_QUALITY_RULES["reject_if_temporal_or_sample_ratio_below"]
    accepted = (
        grade in {"A", "B"}
        and temporal_coverage_ratio >= reject_ratio
        and sample_completeness_ratio >= reject_ratio
    )
    penalties = {"A": 0, "B": 15, "C": 40, "D": 70}
    return {
        "quality_grade": grade,
        "quality_score": 100 - penalties[grade],
        "quality_flags": "|".join(unique_flags),
        "usable_for_window_inference": bool(accepted),
        "trusted_output": "可进入模型（本报告不输出睡眠阶段）" if accepted else "暂不判定",
        **metrics,
    }


__all__ = [
    "ACTIVE_EXG_COLUMNS",
    "ACTIVE_GUI_CHANNELS",
    "OpenBCIHeader",
    "OpenBCISegment",
    "REAL_OPENBCI_QUALITY_RULES",
    "audit_real_openbci_window",
    "clip_intervals",
    "combine_segment_data",
    "experiment_start_timestamp",
    "find_gaps",
    "format_timestamp",
    "interval_coverage",
    "load_event_log",
    "load_experiment_config",
    "load_window_log",
    "merge_intervals",
    "parse_openbci_header",
    "read_openbci_segment",
    "stage_quality_table",
    "stages_from_config",
    "window_integrity_table",
]
