"""Generate formal reports for the SX real OpenBCI experiments."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import json
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

from ..realtime.real_openbci_data import (
    ACTIVE_EXG_COLUMNS,
    ACTIVE_GUI_CHANNELS,
    OpenBCISegment,
    REAL_OPENBCI_QUALITY_RULES,
    clip_intervals,
    combine_segment_data,
    experiment_start_timestamp,
    find_gaps,
    format_timestamp,
    interval_coverage,
    load_event_log,
    load_experiment_config,
    load_window_log,
    read_openbci_segment,
    stage_quality_table,
    stages_from_config,
    window_integrity_table,
)


FORMAL_TEN_FILE = "OpenBCI-RAW-2026-06-27_13-59-52.txt"
DEBUG_TEN_FILE = "OpenBCI-RAW-2026-06-27_13-58-24.txt"
REQUIRED_QUALITY_FLAGS = [
    "line_noise",
    "strong_line_noise",
    "baseline_drift",
    "channel_saturation",
    "flatline",
    "motion_artifact",
    "data_dropout",
    "abnormal_amplitude",
    "zero_variance_channel",
    "effective_data_ratio_insufficient",
]
QUALITY_RULES = REAL_OPENBCI_QUALITY_RULES


def generate_real_openbci_reports(logs_root: str | Path, output_root: str | Path) -> dict:
    logs_root = Path(logs_root)
    output_root = Path(output_root)
    ten_dir, sixty_dir = discover_sx_experiment_dirs(logs_root)
    output_root.mkdir(parents=True, exist_ok=True)
    ten_result = generate_ten_minute_report(ten_dir, output_root / "ten_min_quality_calibration")
    sixty_result = generate_sixty_minute_report(sixty_dir, output_root / "sixty_min_continuous_recording")
    readme = output_root / "REAL_DATA_README.md"
    readme.write_text(_real_data_readme(ten_result, sixty_result), encoding="utf-8")
    return {"ten_minute": ten_result, "sixty_minute": sixty_result, "readme": str(readme)}


def discover_sx_experiment_dirs(logs_root: str | Path) -> tuple[Path, Path]:
    logs_root = Path(logs_root)
    candidates = [path for path in logs_root.iterdir() if path.is_dir()]
    ten = []
    sixty = []
    for directory in candidates:
        try:
            config = load_experiment_config(directory)
            stages = next(
                value
                for value in config.values()
                if isinstance(value, list) and value and isinstance(value[0], dict) and "end_sec" in value[0]
            )
            duration = max(float(stage["end_sec"]) for stage in stages)
        except (FileNotFoundError, StopIteration, ValueError, json.JSONDecodeError):
            continue
        if abs(duration - 600.0) < 1.0:
            ten.append(directory)
        elif abs(duration - 3600.0) < 1.0:
            sixty.append(directory)
    if len(ten) != 1 or len(sixty) != 1:
        raise ValueError(f"expected one 10-minute and one 60-minute SX directory; found {ten=} {sixty=}")
    return ten[0], sixty[0]


def generate_ten_minute_report(input_dir: str | Path, output_dir: str | Path) -> dict:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    formal_path = input_dir / FORMAL_TEN_FILE
    debug_path = input_dir / DEBUG_TEN_FILE
    if not formal_path.exists() or not debug_path.exists():
        raise FileNotFoundError("10-minute formal/debug OpenBCI files were not found")
    formal = read_openbci_segment(formal_path)
    debug = read_openbci_segment(debug_path)
    events = load_event_log(input_dir)
    config = load_experiment_config(input_dir)
    config_meta = _config_metadata(config)
    experiment_start = experiment_start_timestamp(events)
    experiment_end = experiment_start + 600.0
    stages = stages_from_config(config, experiment_start)
    data = combine_segment_data([formal])
    stage_table = stage_quality_table(data, [formal], stages)
    stage_table.to_csv(output_dir / "window_or_stage_quality.csv", index=False, encoding="utf-8-sig")
    intervals = [(formal.start_timestamp, formal.end_timestamp)]
    coverage_sec = interval_coverage(intervals, experiment_start, experiment_end)
    formal_rows = data[(data["timestamp"] >= experiment_start) & (data["timestamp"] < experiment_end)]
    sample_ratio = min(1.0, len(formal_rows) / (600.0 * formal.header.sample_rate))
    gaps = find_gaps(intervals, experiment_start, experiment_end)
    alignment_offset = formal.start_timestamp - experiment_start
    event_integrity = _event_log_integrity(events, 600.0)
    active = sorted(set(formal.active_exg_channels))
    inactive = sorted(set(formal.inactive_exg_channels))
    figures = {
        "full_waveform": _plot_ten_full_waveform(formal_rows, experiment_start, figures_dir / "full_dual_channel_waveform.png"),
        "stage_waveforms": _plot_stage_waveforms(data, stages, figures_dir / "stage_waveforms.png"),
        "stage_spectra": _plot_stage_spectra(data, stages, formal.header.sample_rate, figures_dir / "stage_spectra.png"),
        "quality_timeline": _plot_stage_quality(stage_table, figures_dir / "stage_quality_timeline.png"),
        "artifact_annotations": _plot_artifact_annotations(formal_rows, stages, experiment_start, figures_dir / "artifact_stage_annotations.png"),
    }
    summary = {
        "report_title": "10分钟前额双导脑电质量标定报告",
        "real_data": True,
        "device": formal.header.board,
        "experiment": "Fp1/Fp2 前额双导脑电质量标定实验",
        **config_meta,
        "formal_data_file": formal_path.name,
        "debug_short_file": debug_path.name,
        "debug_file_role": "调试短文件,不用于正式报告主体分析",
        "sample_rate_hz": formal.header.sample_rate,
        "declared_exg_channels": formal.header.number_of_channels,
        "effective_channels": list(ACTIVE_GUI_CHANNELS),
        "effective_exg_columns": list(ACTIVE_EXG_COLUMNS),
        "detected_active_exg_channels": active,
        "inactive_exg_channels": inactive,
        "unexpected_nonzero_inactive_channels": formal.unexpected_nonzero_inactive_channels,
        "experiment_start": format_timestamp(experiment_start),
        "experiment_end": format_timestamp(experiment_end),
        "data_start": format_timestamp(formal.start_timestamp),
        "data_end": format_timestamp(formal.end_timestamp),
        "event_to_data_start_offset_sec": alignment_offset,
        "formal_duration_sec": 600.0,
        "temporal_coverage_sec": coverage_sec,
        "coverage_ratio": coverage_sec / 600.0,
        "sample_count_in_formal_interval": int(len(formal_rows)),
        "sample_completeness_ratio": sample_ratio,
        "missing_intervals": [_gap_row(left, right) for left, right in gaps],
        "event_log_integrity": event_integrity,
        "quality_grade_counts": dict(Counter(stage_table["quality_grade"])),
        "artifact_detection_counts": _flag_counts(stage_table),
        "quality_rules": QUALITY_RULES,
        "rejected_stages": stage_table.loc[~stage_table["usable_for_window_inference"], "stage_name"].tolist(),
        "figures": {key: str(Path("figures") / value.name) for key, value in figures.items()},
        "statement_boundary": [
            "真实 OpenBCI Cyton Fp1/Fp2 双导采集链路可用",
            "用于信号质量审计、伪迹识别、可信拒识和自动报告",
            "不用于睡眠分期准确率验证",
            "不声称 Alpha 验证成功",
        ],
    }
    _write_json(summary, output_dir / "summary.json")
    _write_manifest(input_dir, output_dir / "copied_inputs_manifest.csv", {formal_path.name: "formal_raw", debug_path.name: "debug_short_raw"})
    report_md_body = _ten_markdown(summary, stage_table, formal, debug)
    report_md = report_md_body + _figures_markdown(figures)
    (output_dir / "report.md").write_text(report_md, encoding="utf-8")
    (output_dir / "report.html").write_text(_render_html(summary["report_title"], report_md_body, stage_table, figures), encoding="utf-8")
    return {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "report_html": str(output_dir / "report.html"),
        "report_md": str(output_dir / "report.md"),
        "summary": summary,
    }


def generate_sixty_minute_report(input_dir: str | Path, output_dir: str | Path) -> dict:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    raw_paths = sorted(input_dir.glob("OpenBCI-RAW-*.txt"))
    if len(raw_paths) < 2:
        raise ValueError("60-minute experiment must contain multiple OpenBCI segments")
    segments = [read_openbci_segment(path) for path in raw_paths]
    data = combine_segment_data(segments)
    events = load_event_log(input_dir)
    window_log = load_window_log(input_dir)
    config = load_experiment_config(input_dir)
    config_meta = _config_metadata(config)
    experiment_start = experiment_start_timestamp(events)
    experiment_end = experiment_start + 3600.0
    stages = stages_from_config(config, experiment_start)
    event_integrity = _event_log_integrity(events, 3600.0)
    window_log_integrity = _window_log_integrity(window_log, 120, 3600.0)
    window_table = window_integrity_table(data, segments, experiment_start, window_log)
    window_table.to_csv(output_dir / "thirty_sec_window_integrity.csv", index=False, encoding="utf-8-sig")
    intervals = [(segment.start_timestamp, segment.end_timestamp) for segment in segments]
    clipped = clip_intervals(intervals, experiment_start, experiment_end)
    gaps = find_gaps(intervals, experiment_start, experiment_end)
    coverage_sec = sum(right - left for left, right in clipped)
    gap_table = pd.DataFrame([_gap_row(left, right, index + 1) for index, (left, right) in enumerate(gaps)])
    gap_table.to_csv(output_dir / "data_gap_table.csv", index=False, encoding="utf-8-sig")
    segment_table = pd.DataFrame([segment.summary() for segment in segments])
    segment_table["formal_interval_coverage_sec"] = [
        interval_coverage([(segment.start_timestamp, segment.end_timestamp)], experiment_start, experiment_end)
        for segment in segments
    ]
    segment_table.to_csv(output_dir / "file_segments_table.csv", index=False, encoding="utf-8-sig")
    stage_stats = _stage_window_statistics(window_table, stages)
    figures = {
        "coverage_timeline": _plot_coverage_timeline(clipped, gaps, experiment_start, experiment_end, figures_dir / "data_coverage_timeline.png"),
        "window_integrity": _plot_window_integrity(window_table, figures_dir / "window_integrity.png"),
        "waveform_overview": _plot_sixty_waveform(data, experiment_start, experiment_end, figures_dir / "dual_channel_waveform_overview.png"),
        "quality_trend": _plot_window_quality(window_table, figures_dir / "quality_grade_trend.png"),
        "rejection_distribution": _plot_rejections(window_table, figures_dir / "rejected_windows.png"),
        "gap_intervals": _plot_gap_intervals(gaps, experiment_start, figures_dir / "data_gap_intervals.png"),
    }
    file_gap_affected = window_table.loc[window_table["coverage_ratio"] < 0.999, "window_index"].astype(int).tolist()
    sample_incomplete = window_table.loc[
        window_table["sample_completeness_ratio"] < 0.95, "window_index"
    ].astype(int).tolist()
    affected = sorted(set(file_gap_affected) | set(sample_incomplete))
    rejected = window_table.loc[~window_table["usable_for_window_inference"], "window_index"].astype(int).tolist()
    formal_data = data[(data["timestamp"] >= experiment_start) & (data["timestamp"] < experiment_end)]
    total_expected_samples = int(round(3600.0 * segments[0].header.sample_rate))
    summary = {
        "report_title": "60分钟午睡/闭眼休息连续采集报告",
        "real_data": True,
        "device": segments[0].header.board,
        "experiment": "Fp1/Fp2 前额双导连续采集验证",
        **config_meta,
        "sample_rate_hz": segments[0].header.sample_rate,
        "declared_exg_channels": segments[0].header.number_of_channels,
        "effective_channels": list(ACTIVE_GUI_CHANNELS),
        "effective_exg_columns": list(ACTIVE_EXG_COLUMNS),
        "segment_count": len(segments),
        "segment_files": [path.name for path in raw_paths],
        "unexpected_nonzero_inactive_channels_by_segment": {
            segment.path.name: segment.unexpected_nonzero_inactive_channels
            for segment in segments
            if segment.unexpected_nonzero_inactive_channels
        },
        "experiment_start": format_timestamp(experiment_start),
        "experiment_end": format_timestamp(experiment_end),
        "formal_duration_sec": 3600.0,
        "temporal_coverage_sec": coverage_sec,
        "coverage_ratio": coverage_sec / 3600.0,
        "missing_duration_sec": 3600.0 - coverage_sec,
        "sample_count_in_formal_interval": int(len(formal_data)),
        "expected_sample_count": total_expected_samples,
        "sample_completeness_ratio": min(1.0, len(formal_data) / total_expected_samples),
        "gap_count": len(gaps),
        "event_log_integrity": event_integrity,
        "window_log_integrity": window_log_integrity,
        "affected_window_indices": affected,
        "file_gap_affected_window_indices": file_gap_affected,
        "sample_incomplete_window_indices": sample_incomplete,
        "rejected_window_indices": rejected,
        "quality_grade_counts": dict(Counter(window_table["quality_grade"])),
        "artifact_detection_counts": _flag_counts(window_table),
        "quality_rules": QUALITY_RULES,
        "stage_quality_statistics": stage_stats,
        "figures": {key: str(Path("figures") / value.name) for key, value in figures.items()},
        "statement_boundary": [
            "完成 60 分钟午睡/闭眼休息场景真实采集流程",
            "OpenBCI 原始数据大部分覆盖正式实验区间,但存在短暂丢包和数据中断",
            "低覆盖或严重异常窗口输出暂不判定",
            "用于连续采集、质量审计、30 秒滑窗、可信拒识和自动报告验证",
            "无 PSG 和专家 30 秒睡眠标签,不作为睡眠分期准确率依据",
        ],
    }
    _write_json(summary, output_dir / "summary.json")
    _write_manifest(input_dir, output_dir / "copied_inputs_manifest.csv", {path.name: "continuous_raw_segment" for path in raw_paths})
    report_md_body = _sixty_markdown(summary, segment_table, gap_table, window_table, stage_stats)
    report_md = report_md_body + _figures_markdown(figures)
    (output_dir / "report.md").write_text(report_md, encoding="utf-8")
    (output_dir / "report.html").write_text(_render_html(summary["report_title"], report_md_body, window_table, figures), encoding="utf-8")
    return {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "report_html": str(output_dir / "report.html"),
        "report_md": str(output_dir / "report.md"),
        "summary": summary,
    }


def _plot_ten_full_waveform(data: pd.DataFrame, start: float, output: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True, constrained_layout=True)
    sampled = _decimate_frame(data, 12_000)
    time_min = (sampled["timestamp"].to_numpy() - start) / 60.0
    for ax, column, label, color in zip(axes, ["ch2_uv", "ch7_uv"], ACTIVE_GUI_CHANNELS, ["#176B87", "#B04A3A"]):
        values = sampled[column].to_numpy(float)
        ax.plot(time_min, values - np.nanmedian(values), linewidth=0.5, color=color)
        ax.set_ylabel(f"{label} (uV, centered)")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Experiment time (min)")
    fig.suptitle("Real OpenBCI Fp1/Fp2 full recording waveform")
    return _save_figure(fig, output)


def _plot_stage_waveforms(data: pd.DataFrame, stages: Sequence[dict], output: Path) -> Path:
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)
    for ax, stage in zip(axes.flat, stages):
        selected = data[(data["timestamp"] >= stage["absolute_start"]) & (data["timestamp"] < stage["absolute_end"])]
        selected = selected.iloc[: min(len(selected), 1250)]
        if selected.empty:
            ax.set_axis_off()
            continue
        t = np.arange(len(selected)) / 250.0
        for column, color, label in zip(["ch2_uv", "ch7_uv"], ["#176B87", "#B04A3A"], ["Ch2/Fp1", "Ch7/Fp2"]):
            values = selected[column].to_numpy(float)
            ax.plot(t, values - np.median(values), linewidth=0.6, color=color, label=label)
        ax.set_title(str(stage["key"]))
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Representative waveforms by protocol stage")
    return _save_figure(fig, output)


def _plot_stage_spectra(data: pd.DataFrame, stages: Sequence[dict], sfreq: float, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for stage in stages:
        selected = data[(data["timestamp"] >= stage["absolute_start"]) & (data["timestamp"] < stage["absolute_end"])]
        if len(selected) < 64:
            continue
        signal = selected[["ch2_uv", "ch7_uv"]].to_numpy(float).mean(axis=1)
        signal -= np.median(signal)
        freqs, psd = welch(signal, fs=sfreq, nperseg=min(len(signal), int(sfreq * 8)))
        mask = (freqs >= 0.3) & (freqs <= 60)
        ax.semilogy(freqs[mask], psd[mask], linewidth=0.9, label=str(stage["key"]))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (uV^2/Hz)")
    ax.set_title("Stage spectra for quality comparison (not an Alpha validation claim)")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    return _save_figure(fig, output)


def _plot_stage_quality(table: pd.DataFrame, output: Path) -> Path:
    values = table["quality_grade"].map({"A": 4, "B": 3, "C": 2, "D": 1}).to_numpy()
    colors = table["quality_grade"].map({"A": "#2A9D5B", "B": "#D4A72C", "C": "#E67E22", "D": "#C0392B"})
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax.bar(np.arange(len(table)), values, color=colors)
    ax.set_xticks(np.arange(len(table)), table["stage_key"], rotation=30, ha="right")
    ax.set_yticks([1, 2, 3, 4], ["D", "C", "B", "A"])
    ax.set_title("Protocol-stage quality grade")
    ax.grid(True, axis="y", alpha=0.25)
    return _save_figure(fig, output)


def _plot_artifact_annotations(data: pd.DataFrame, stages: Sequence[dict], start: float, output: Path) -> Path:
    sampled = _decimate_frame(data, 10_000)
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    t = sampled["timestamp"].to_numpy() - start
    values = sampled["ch2_uv"].to_numpy(float)
    ax.plot(t, values - np.median(values), linewidth=0.5, color="#176B87")
    for stage in stages:
        if stage["key"] in {"blink", "jaw", "head_turn", "cable_move"}:
            ax.axvspan(stage["start_sec"], stage["end_sec"], alpha=0.25, label=stage["key"])
    ax.set_xlabel("Experiment time (s)")
    ax.set_ylabel("Ch2/Fp1 (uV, centered)")
    ax.set_title("Protocol-induced artifact intervals")
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=8)
    return _save_figure(fig, output)


def _plot_coverage_timeline(intervals, gaps, start, end, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14, 2.8), constrained_layout=True)
    for left, right in intervals:
        ax.broken_barh([((left - start) / 60, (right - left) / 60)], (0.55, 0.35), facecolors="#2A9D5B")
    for left, right in gaps:
        ax.broken_barh([((left - start) / 60, (right - left) / 60)], (0.1, 0.35), facecolors="#C0392B")
    ax.set_xlim(0, (end - start) / 60)
    ax.set_yticks([0.275, 0.725], ["Missing", "Recorded"])
    ax.set_xlabel("Experiment time (min)")
    ax.set_title("Real data temporal coverage")
    ax.grid(True, axis="x", alpha=0.25)
    return _save_figure(fig, output)


def _plot_window_integrity(table: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
    colors = np.where(table["effective_data_ratio"] >= 0.8, "#2A9D5B", "#C0392B")
    ax.bar(table["window_index"], table["effective_data_ratio"], width=0.9, color=colors)
    ax.axhline(0.8, color="#222222", linestyle="--", linewidth=1, label="rejection threshold")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("30-second window index")
    ax.set_ylabel("Effective data ratio")
    ax.set_title("30-second window integrity (minimum of temporal and sample coverage)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    return _save_figure(fig, output)


def _plot_sixty_waveform(data: pd.DataFrame, start: float, end: float, output: Path) -> Path:
    selected = data[(data["timestamp"] >= start) & (data["timestamp"] < end)]
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True, constrained_layout=True)
    for ax, column, label, color in zip(axes, ["ch2_uv", "ch7_uv"], ACTIVE_GUI_CHANNELS, ["#176B87", "#B04A3A"]):
        center = np.nanmedian(selected[column].to_numpy(float))
        for _, segment in selected.groupby("source_file", sort=False):
            sampled = _decimate_frame(segment, max(500, 18_000 // max(1, selected["source_file"].nunique())))
            time_min = (sampled["timestamp"].to_numpy() - start) / 60.0
            values = sampled[column].to_numpy(float)
            ax.plot(time_min, values - center, linewidth=0.45, color=color)
        ax.set_ylabel(f"{label} (uV)")
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("Experiment time (min)")
    fig.suptitle("Real OpenBCI Fp1/Fp2 waveform overview; gaps are not interpolated")
    return _save_figure(fig, output)


def _plot_window_quality(table: pd.DataFrame, output: Path) -> Path:
    values = table["quality_grade"].map({"A": 4, "B": 3, "C": 2, "D": 1})
    fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
    ax.step(table["window_index"], values, where="mid", color="#176B87")
    ax.scatter(table["window_index"], values, c=table["quality_grade"].map({"A": "#2A9D5B", "B": "#D4A72C", "C": "#E67E22", "D": "#C0392B"}), s=18)
    ax.set_yticks([1, 2, 3, 4], ["D", "C", "B", "A"])
    ax.set_xlabel("30-second window index")
    ax.set_title("Signal quality grade trend")
    ax.grid(True, alpha=0.25)
    return _save_figure(fig, output)


def _plot_rejections(table: pd.DataFrame, output: Path) -> Path:
    rejected = table.loc[~table["usable_for_window_inference"], "window_index"]
    fig, ax = plt.subplots(figsize=(14, 3.5), constrained_layout=True)
    ax.scatter(rejected, np.ones(len(rejected)), marker="x", s=45, color="#C0392B")
    ax.set_xlim(0, 121)
    ax.set_yticks([1], ["Reject"])
    ax.set_xlabel("30-second window index")
    ax.set_title("Windows marked as temporarily indeterminate")
    ax.grid(True, axis="x", alpha=0.25)
    return _save_figure(fig, output)


def _plot_gap_intervals(gaps, start, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14, 3.5), constrained_layout=True)
    for index, (left, right) in enumerate(gaps, start=1):
        ax.barh(index, (right - left) / 60.0, left=(left - start) / 60.0, height=0.65, color="#C0392B")
    ax.set_xlabel("Experiment time (min)")
    ax.set_ylabel("Gap index")
    ax.set_title("Data interruption intervals")
    ax.grid(True, axis="x", alpha=0.25)
    return _save_figure(fig, output)


def _stage_window_statistics(window_table: pd.DataFrame, stages: Sequence[dict]) -> list[dict]:
    rows = []
    for stage in stages:
        subset = window_table[
            (window_table["window_start_sec"] >= float(stage["start_sec"]))
            & (window_table["window_end_sec"] <= float(stage["end_sec"]))
        ]
        rows.append(
            {
                "stage_key": stage["key"],
                "stage_name": stage["name"],
                "window_count": int(len(subset)),
                "mean_coverage_ratio": float(subset["coverage_ratio"].mean()) if len(subset) else 0.0,
                "mean_sample_completeness_ratio": float(subset["sample_completeness_ratio"].mean()) if len(subset) else 0.0,
                "grade_counts": dict(Counter(subset["quality_grade"])),
                "rejected_window_count": int((~subset["usable_for_window_inference"]).sum()),
            }
        )
    return rows


def _ten_markdown(summary: dict, table: pd.DataFrame, formal: OpenBCISegment, debug: OpenBCISegment) -> str:
    comparisons = table[table["stage_key"].isin(["eo_1", "ec_1", "eo_2", "ec_2"])][
        ["stage_name", "quality_grade", "ch2_std_uv", "ch7_std_uv", "ch2_line_noise_ratio", "ch7_line_noise_ratio", "quality_flags"]
    ]
    artifact = table[table["protocol_artifact_stage"]][
        ["stage_name", "quality_grade", "quality_flags", "usable_for_window_inference", "trusted_output"]
    ]
    return f"""# 10分钟前额双导脑电质量标定报告

## 实验基本信息

- 数据性质:OpenBCI Cyton + OpenBCI GUI 真实采集数据。
- 实验定位:Fp1/Fp2 前额双导脑电质量标定,不是 O1/O2 枕区实验,不主张闭眼 Alpha 验证成功。
- 被试编号:{summary['subject_id']};操作者:{summary['operator']};运行模式:{summary['run_mode']};点位方案:{summary['montage']}。
- 正式原始文件:`{Path(summary['formal_data_file']).name}`。
- 调试短文件:`{Path(summary['debug_short_file']).name}`,时长 {debug.duration_sec:.3f} 秒,不进入正式主体分析。
- 设备:`{formal.header.board}`;采样率:{formal.header.sample_rate:.0f} Hz;声明 EXG 通道数:{formal.header.number_of_channels}。
- 有效通道:GUI Ch2/Fp1 与 Ch7/Fp2,对应 TXT 的 `EXG Channel 1`、`EXG Channel 6`。
- 关闭通道:{', '.join(summary['inactive_exg_channels']) or '无'};关闭通道不参与质量评分。
- 停用通道非零活动:{', '.join(summary['unexpected_nonzero_inactive_channels']) or '未发现'}。

## 时间对齐与覆盖

- 事件区间:{summary['experiment_start']} 至 {summary['experiment_end']},600 秒。
- 正式数据区间:{summary['data_start']} 至 {summary['data_end']}。
- 数据相对事件开始晚 {summary['event_to_data_start_offset_sec']:.3f} 秒。
- 事件日志共 {summary['event_log_integrity']['row_count']} 行;首尾时长校验:{summary['event_log_integrity']['complete']}。
- 正式区间时间覆盖 {summary['temporal_coverage_sec']:.3f} 秒,覆盖率 {summary['coverage_ratio']:.3%}。
- 样本完整率 {summary['sample_completeness_ratio']:.3%};逐行 GUI 时间戳抖动不被单独解释为文件中断。

## 各阶段质量

{_markdown_table(table[['stage_name','start_sec','end_sec','coverage_ratio','sample_completeness_ratio','quality_grade','quality_flags','trusted_output']])}

## 睁眼/闭眼阶段质量对比

{_markdown_table(comparisons)}

以上仅比较前额双导信号幅值、工频比例和质量等级,不作 Alpha 增强结论。

## 诱发伪迹阶段

{_markdown_table(artifact)}

眨眼、咬牙、转头和动线是按事件日志主动诱发的质量案例,统一作为 C 级质量案例并拒绝进入可信推理历史。

## 异常检测汇总

{_markdown_table(pd.DataFrame([{'异常类型': key, '阶段数': value} for key, value in summary['artifact_detection_counts'].items()]))}

## 质量规则

- 时间覆盖率或样本完整率低于 80%:D 级并暂不判定。
- 强工频比阈值:0.5;一般工频比阈值:0.2。
- 滤波后 99% 绝对幅值超过 500 µV 或差分 99% 超过 50 µV:运动伪迹候选。
- 协议规定的眨眼、咬牙、转头、动线阶段固定作为 C 级质量案例,不进入可信历史。

## 结论与边界

1. 真实 OpenBCI Cyton 对 Fp1/Fp2 前额双导的采集、文件保存和时间对齐链路可用。
2. 本实验可用于工频、漂移、异常幅值、运动伪迹、平线、饱和和中断检测,以及主动拒识验证。
3. C/D 级阶段输出"暂不判定";A/B 级阶段仅表示信号可进入后续模型,本报告不输出睡眠阶段。
4. 本实验没有 PSG 和专家 30 秒睡眠标签,不用于睡眠分期准确率验证。
5. 不声称 Alpha 验证成功;公开 Sleep-EDF/ISRUC 才用于分期准确率验证。
"""


def _sixty_markdown(summary: dict, segment_table: pd.DataFrame, gap_table: pd.DataFrame, windows: pd.DataFrame, stage_stats: list[dict]) -> str:
    affected = windows[windows["window_index"].isin(summary["affected_window_indices"])][
        ["window_index", "stage", "coverage_ratio", "sample_completeness_ratio", "effective_data_ratio", "quality_grade", "quality_flags", "trusted_output"]
    ]
    inactive_activity = ";".join(
        f"{file_name}: {', '.join(channels)}"
        for file_name, channels in summary["unexpected_nonzero_inactive_channels_by_segment"].items()
    ) or "未发现"
    return f"""# 60分钟午睡/闭眼休息连续采集报告

## 实验基本信息

- 数据性质:OpenBCI Cyton + OpenBCI GUI 真实采集数据。
- 点位:Fp1/Fp2 前额双导;有效通道为 GUI Ch2、Ch7。
- 被试编号:{summary['subject_id']};操作者:{summary['operator']};运行模式:{summary['run_mode']};点位方案:{summary['montage']}。
- 正式实验区间:{summary['experiment_start']} 至 {summary['experiment_end']},3600 秒,理论 30 秒窗口 120 个。
- 原始数据由 {summary['segment_count']} 个文件片段组成,不对片段间缺口进行插值。
- 事件日志完整性:{summary['event_log_integrity']['complete']};30 秒窗口日志完整性:{summary['window_log_integrity']['complete']}({summary['window_log_integrity']['row_count']} 行)。
- 配置禁用通道始终不参与质量评分;短片段中的非零禁用通道活动单独记录在分段表与 summary 中。
- 禁用通道异常活动:{inactive_activity}。这些通道未进入双导波形、质量评分或拒识判断。

## 原始文件片段

{_markdown_table(segment_table[['file_name','start_time','end_time','duration_sec','row_count','formal_interval_coverage_sec']])}

## 数据覆盖与中断

- 时间覆盖:{summary['temporal_coverage_sec']:.3f} 秒,覆盖率 {summary['coverage_ratio']:.3%}。
- 缺失时长:{summary['missing_duration_sec']:.3f} 秒,共 {summary['gap_count']} 个文件级缺口。
- 样本完整率:{summary['sample_completeness_ratio']:.3%},与时间覆盖率分开报告。

{_markdown_table(gap_table)}

## 受影响窗口

文件级时间缺口直接影响窗口:{', '.join(map(str, summary['file_gap_affected_window_indices']))}。

样本完整率低于 95% 的窗口:{', '.join(map(str, summary['sample_incomplete_window_indices']))}。

两类问题合并后的受影响窗口:{', '.join(map(str, summary['affected_window_indices']))}。

{_markdown_table(affected)}

所有覆盖不足 80%、严重中断、平线、饱和或严重运动伪迹窗口均输出"暂不判定"。

## 分阶段质量统计

{_markdown_table(pd.DataFrame(stage_stats))}

## 异常检测汇总

{_markdown_table(pd.DataFrame([{'异常类型': key, '窗口数': value} for key, value in summary['artifact_detection_counts'].items()]))}

## 质量规则

- 时间覆盖率或样本完整率低于 80%:D 级并暂不判定。
- 文件级覆盖与样本完整率分别计算,有效数据比例取两者较小值。
- 强工频、严重运动或异常幅值为 C 级并暂不判定;平线、饱和、零方差为 D 级。

## 结论与边界

1. 已完成 60 分钟午睡/闭眼休息场景采集,事件日志和 120 个 30 秒窗口日志完整。
2. OpenBCI 原始数据覆盖大部分正式区间,但中间存在真实文件级缺口,不能写成"全程无丢包"。
3. 缺失和严重低质量窗口已标为"暂不判定",不会进入可信历史特征。
4. 该实验验证连续采集、文件兼容、质量审计、30 秒滑窗、数据中断检测、可信拒识和自动报告。
5. "自然午睡 / 闭眼休息"是实验场景名称;无 PSG、无专家睡眠标签,不作为睡眠分期准确率依据,不能声称被试已进入睡眠或验证了睡眠分期准确率。
6. 公开 Sleep-EDF/ISRUC 才用于睡眠分期准确率验证。
"""


def _render_html(title: str, markdown_text: str, table: pd.DataFrame, figures: dict[str, Path]) -> str:
    sections = markdown_text.split("\n")
    body = []
    in_list = False
    index = 0
    while index < len(sections):
        line = sections[index]
        if line.startswith("# "):
            body.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            if in_list:
                body.append("</ul>")
                in_list = False
            body.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("- "):
            if not in_list:
                body.append("<ul>")
                in_list = True
            body.append(f"<li>{line[2:]}</li>")
        elif line.startswith("|"):
            table_lines = []
            while index < len(sections) and sections[index].startswith("|"):
                table_lines.append(sections[index])
                index += 1
            body.append(_markdown_table_to_html(table_lines))
            continue
        elif line:
            if in_list:
                body.append("</ul>")
                in_list = False
            body.append(f"<p>{line}</p>")
        index += 1
    if in_list:
        body.append("</ul>")
    figure_html = "\n".join(
        f'<figure><img src="figures/{path.name}" alt="{key}"><figcaption>{key}</figcaption></figure>'
        for key, path in figures.items()
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{title}</title>
<style>
body{{font-family:Arial,"Microsoft YaHei",sans-serif;margin:28px;color:#1f2933;line-height:1.55;max-width:1400px}}
h1,h2{{color:#16324f}} table{{border-collapse:collapse;width:100%;font-size:12px;margin:12px 0 24px}}
th,td{{border:1px solid #ccd5df;padding:5px 7px;text-align:left}} th{{background:#eef3f8;position:sticky;top:0}}
figure{{margin:24px 0}} img{{max-width:100%;border:1px solid #ccd5df}} .notice{{background:#fff7df;border-left:4px solid #b7791f;padding:12px}}
</style></head><body>
{''.join(body)}
<h2>完整质量明细</h2>
<div style="overflow-x:auto">{table.to_html(index=False, border=0)}</div>
<h2>图表</h2>{figure_html}
<p class="notice">本报告基于真实 OpenBCI 数据,但无 PSG 与专家睡眠标签;不得用于声称睡眠分期准确率或被试已进入睡眠。</p>
</body></html>"""


def _write_manifest(input_dir: Path, output: Path, role_overrides: dict[str, str]) -> None:
    rows = []
    for path in sorted(item for item in input_dir.iterdir() if item.is_file()):
        role = role_overrides.get(path.name, _infer_role(path))
        used = role in {
            "formal_raw",
            "continuous_raw_segment",
            "event_log",
            "thirty_second_window_log",
            "experiment_config",
        }
        rows.append(
            {
                "source_path": path.name,
                "file_name": path.name,
                "role": role,
                "used_in_formal_analysis": used,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "copy_policy": "referenced_in_place; original not modified or duplicated",
            }
        )
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")


def _infer_role(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "experiment_config"
    if suffix == ".csv" and "30" in path.name:
        return "thirty_second_window_log"
    if suffix == ".csv":
        return "event_log"
    if suffix == ".png":
        return "protocol_timeline"
    if "拍摄清单" in path.name:
        return "capture_checklist"
    if "实验说明" in path.name:
        return "experiment_note"
    if suffix == ".txt" and path.stat().st_size == 0:
        return "data_note"
    if suffix == ".txt":
        return "experiment_note_or_checklist"
    return "supporting_input"


def _config_metadata(config: dict) -> dict:
    return {
        "subject_id": str(config.get("被试编号", "unknown")),
        "operator": str(config.get("操作者", "unknown")),
        "run_mode": str(config.get("运行模式", "unknown")),
        "montage": str(config.get("点位方案", "Fp1Fp2")),
    }


def _event_log_integrity(events: pd.DataFrame, expected_duration_sec: float) -> dict:
    relative = pd.to_numeric(events.iloc[:, 1], errors="coerce")
    first = float(relative.dropna().iloc[0]) if relative.notna().any() else float("nan")
    last = float(relative.dropna().iloc[-1]) if relative.notna().any() else float("nan")
    complete = bool(np.isfinite(first) and np.isfinite(last) and abs(first) <= 0.1 and abs(last - expected_duration_sec) <= 0.2)
    return {"row_count": int(len(events)), "first_relative_sec": first, "last_relative_sec": last, "complete": complete}


def _window_log_integrity(window_log: pd.DataFrame, expected_windows: int, expected_duration_sec: float) -> dict:
    indices = pd.to_numeric(window_log.iloc[:, 0], errors="coerce").dropna().astype(int).tolist()
    starts = pd.to_numeric(window_log.iloc[:, 1], errors="coerce")
    ends = pd.to_numeric(window_log.iloc[:, 2], errors="coerce")
    expected_indices = list(range(1, expected_windows + 1))
    complete = bool(
        len(window_log) == expected_windows
        and indices == expected_indices
        and len(starts)
        and abs(float(starts.iloc[0])) <= 1e-6
        and abs(float(ends.iloc[-1]) - expected_duration_sec) <= 1e-6
    )
    return {
        "row_count": int(len(window_log)),
        "first_index": indices[0] if indices else None,
        "last_index": indices[-1] if indices else None,
        "first_start_sec": float(starts.iloc[0]) if len(starts) else None,
        "last_end_sec": float(ends.iloc[-1]) if len(ends) else None,
        "complete": complete,
    }


def _flag_counts(table: pd.DataFrame) -> dict[str, int]:
    counters = Counter(
        flag
        for text in table["quality_flags"].fillna("").astype(str)
        for flag in text.split("|")
        if flag
    )
    return {flag: int(counters.get(flag, 0)) for flag in REQUIRED_QUALITY_FLAGS}


def _gap_row(left: float, right: float, index: int | None = None) -> dict:
    row = {
        "gap_start": format_timestamp(left),
        "gap_end": format_timestamp(right),
        "missing_duration_sec": float(right - left),
    }
    if index is not None:
        row = {"gap_index": index, **row}
    return row


def _decimate_frame(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    step = max(1, len(frame) // max_points)
    return frame.iloc[::step]


def _save_figure(fig, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(无)"
    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    headers = [str(column) for column in display.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(value).replace("|", "/") for value in row) + " |")
    return "\n".join(lines)


def _figures_markdown(figures: dict[str, Path]) -> str:
    lines = ["\n## 图表\n"]
    for key, path in figures.items():
        lines.append(f"### {key}\n\n![{key}](figures/{path.name})\n")
    return "\n".join(lines)


def _markdown_table_to_html(lines: Sequence[str]) -> str:
    if len(lines) < 2:
        return ""
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    headers = rows[0]
    body_rows = rows[2:] if all(set(cell) <= {"-", ":"} for cell in rows[1]) else rows[1:]
    head = "".join(f"<th>{cell}</th>" for cell in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in body_rows)
    return f'<div style="overflow-x:auto"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _write_json(data: dict, output: Path) -> None:
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _real_data_readme(ten_result: dict, sixty_result: dict) -> str:
    ten = ten_result["summary"]
    sixty = sixty_result["summary"]
    return f"""# SX Real OpenBCI Reports

本目录只包含 2026-06-27 OpenBCI Cyton + OpenBCI GUI 真实实验报告,不把 synthetic 数据作为正式结果。

## 报告

- 10 分钟 Fp1/Fp2 质量标定:[HTML](ten_min_quality_calibration/report.html) / [Markdown](ten_min_quality_calibration/report.md)
- 60 分钟午睡/闭眼休息连续采集:[HTML](sixty_min_continuous_recording/report.html) / [Markdown](sixty_min_continuous_recording/report.md)

10 分钟正式文件:`{Path(ten['formal_data_file']).name}`;调试短文件:`{Path(ten['debug_short_file']).name}`。

60 分钟时间覆盖率:{sixty['coverage_ratio']:.3%};文件级缺失时长:{sixty['missing_duration_sec']:.3f} 秒;文件级缺口窗口:{', '.join(map(str, sixty['file_gap_affected_window_indices']))};全部受影响窗口:{', '.join(map(str, sixty['affected_window_indices']))}。

## 允许写入比赛材料

- 完成真实 OpenBCI Cyton Fp1/Fp2 双导采集链路与文件兼容验证。
- 完成 30 秒窗口完整性、信号质量、数据中断、主动拒识和自动报告验证。
- 10 分钟实验提供睁眼、闭眼及主动诱发伪迹质量案例。
- 60 分钟正式区间大部分有数据覆盖,中间存在明确缺口,受影响窗口已拒识。

## 禁止结论

- 不得声称被试已进入睡眠、已证明睡眠分期准确率或完成 PSG/专家标签验证。
- 不得声称全程无丢包。
- 不得把 10 分钟实验写成 O1/O2 Alpha 验证或声称 Alpha 验证成功。
- 睡眠分期准确率只能由带专家标签的 Sleep-EDF/ISRUC 公开数据验证。
"""


__all__ = [
    "DEBUG_TEN_FILE",
    "FORMAL_TEN_FILE",
    "discover_sx_experiment_dirs",
    "generate_real_openbci_reports",
    "generate_sixty_minute_report",
    "generate_ten_minute_report",
]
