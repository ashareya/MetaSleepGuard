from pathlib import Path
import json

import numpy as np
import pandas as pd

from MetaSleepGuard.realtime.real_openbci_data import (
    ACTIVE_EXG_COLUMNS,
    audit_real_openbci_window,
    combine_segment_data,
    experiment_start_timestamp,
    find_gaps,
    interval_coverage,
    parse_openbci_header,
    read_openbci_segment,
    window_integrity_table,
)
from MetaSleepGuard.reports.real_openbci_reports import generate_real_openbci_reports


def test_openbci_txt_header_parsing(tmp_path):
    path = _write_openbci(tmp_path / "sample.txt", [1000.0, 1000.004])
    header = parse_openbci_header(path)
    assert header.number_of_channels == 8
    assert header.sample_rate == 250
    assert "Cyton" in header.board


def test_multiple_openbci_segments_merge_in_timestamp_order(tmp_path):
    late = read_openbci_segment(_write_openbci(tmp_path / "late.txt", [1010.0, 1010.004]))
    early = read_openbci_segment(_write_openbci(tmp_path / "early.txt", [1000.0, 1000.004]))
    merged = combine_segment_data([late, early])
    assert merged["timestamp"].is_monotonic_increasing
    assert merged.iloc[0]["source_file"] == "early.txt"


def test_event_log_alignment_uses_first_record_time():
    events = pd.DataFrame({"record_time": ["2026-06-27T14:12:00+08:00"], "relative": [0.0]})
    start = experiment_start_timestamp(events)
    assert pd.to_datetime(start, unit="s", utc=True).tz_convert("Asia/Shanghai").hour == 14


def test_thirty_second_window_integrity(tmp_path):
    timestamps = 1000.0 + np.arange(7500) / 250.0
    segment = read_openbci_segment(_write_openbci(tmp_path / "full.txt", timestamps))
    data = combine_segment_data([segment])
    log = pd.DataFrame([[1, 0.0, 30.0, "baseline", 100, "time", ""]])
    table = window_integrity_table(data, [segment], 1000.0, log)
    assert table.iloc[0]["coverage_ratio"] > 0.999
    assert table.iloc[0]["sample_completeness_ratio"] == 1.0
    assert table.iloc[0]["effective_data_ratio"] > 0.999


def test_data_gap_detection():
    intervals = [(0.0, 10.0), (15.0, 20.0)]
    assert interval_coverage(intervals, 0.0, 20.0) == 15.0
    assert find_gaps(intervals, 0.0, 20.0) == [(10.0, 15.0)]


def test_disabled_channels_do_not_replace_effective_channels(tmp_path):
    segment = read_openbci_segment(_write_openbci(tmp_path / "channels.txt", 1000 + np.arange(100) / 250))
    assert set(ACTIVE_EXG_COLUMNS) <= set(segment.active_exg_channels)
    assert "EXG Channel 0" in segment.inactive_exg_channels
    assert "EXG Channel 7" in segment.inactive_exg_channels
    assert not segment.unexpected_nonzero_inactive_channels


def test_fp1_fp2_effective_channel_identification(tmp_path):
    segment = read_openbci_segment(_write_openbci(tmp_path / "fp.txt", 1000 + np.arange(100) / 250))
    assert segment.data.columns.tolist() == ["timestamp", "ch2_uv", "ch7_uv"]
    assert np.std(segment.data["ch2_uv"]) > 0
    assert np.std(segment.data["ch7_uv"]) > 0


def test_low_coverage_window_is_temporarily_indeterminate():
    signal = np.vstack([np.sin(np.linspace(0, 10, 100)), np.cos(np.linspace(0, 10, 100))])
    quality = audit_real_openbci_window(signal, 250, temporal_coverage_ratio=0.5, sample_completeness_ratio=0.5)
    assert quality["quality_grade"] == "D"
    assert quality["trusted_output"] == "暂不判定"
    assert not quality["usable_for_window_inference"]


def test_real_report_generation_smoke(tmp_path):
    logs = tmp_path / "logs"
    ten = logs / "sx_ten"
    sixty = logs / "sx_sixty"
    ten.mkdir(parents=True)
    sixty.mkdir(parents=True)
    start_ten = pd.Timestamp("2026-06-27T13:59:49+08:00").timestamp()
    _write_openbci(ten / "OpenBCI-RAW-2026-06-27_13-59-52.txt", start_ten + np.arange(100) / 250)
    _write_openbci(ten / "OpenBCI-RAW-2026-06-27_13-58-24.txt", start_ten - 60 + np.arange(20) / 250)
    _write_protocol_files(ten, start_ten, duration=600, windows=None)
    start_sixty = pd.Timestamp("2026-06-27T14:12:00+08:00").timestamp()
    _write_openbci(sixty / "OpenBCI-RAW-2026-06-27_14-11-54.txt", start_sixty + np.arange(100) / 250)
    _write_openbci(sixty / "OpenBCI-RAW-2026-06-27_14-56-19.txt", start_sixty + 1800 + np.arange(100) / 250)
    _write_protocol_files(sixty, start_sixty, duration=3600, windows=[[1, 0, 30, "baseline", 100, "time", ""]])
    output = tmp_path / "reports"
    result = generate_real_openbci_reports(logs, output)
    assert Path(result["ten_minute"]["report_html"]).exists()
    assert Path(result["sixty_minute"]["report_html"]).exists()
    assert (output / "REAL_DATA_README.md").exists()
    sixty_markdown = (output / "sixty_min_continuous_recording" / "report.md").read_text(encoding="utf-8")
    assert "禁用通道异常活动" in sixty_markdown
    assert "不作为睡眠分期准确率依据" in sixty_markdown
    ten_summary = result["ten_minute"]["summary"]
    sixty_summary = result["sixty_minute"]["summary"]
    assert Path(ten_summary["formal_data_file"]).name == ten_summary["formal_data_file"]
    assert all(Path(path).name == path for path in sixty_summary["segment_files"])
    assert all(not Path(path).is_absolute() for path in ten_summary["figures"].values())
    assert all(not Path(path).is_absolute() for path in sixty_summary["figures"].values())
    manifest = pd.read_csv(output / "ten_min_quality_calibration" / "copied_inputs_manifest.csv")
    assert all(Path(path).name == path for path in manifest["source_path"])


def _write_openbci(path: Path, timestamps) -> Path:
    timestamps = np.asarray(list(timestamps), dtype=float)
    sample = np.arange(len(timestamps), dtype=float)
    frame = pd.DataFrame({"Sample Index": sample % 256})
    for index in range(8):
        if index == 1:
            frame[f"EXG Channel {index}"] = 10 * np.sin(sample / 5) + sample * 0.01
        elif index == 6:
            frame[f"EXG Channel {index}"] = 8 * np.cos(sample / 7) - sample * 0.01
        else:
            frame[f"EXG Channel {index}"] = 0.0
    frame["Timestamp"] = timestamps
    header = "%OpenBCI Raw EXG Data\n%Number of channels = 8\n%Sample Rate = 250 Hz\n%Board = OpenBCI_GUI$BoardCytonSerial\n"
    path.write_text(header + frame.to_csv(index=False), encoding="utf-8")
    return path


def _write_protocol_files(directory: Path, start_timestamp: float, duration: int, windows):
    pd.DataFrame(
        {
            "record_time": [pd.to_datetime(start_timestamp, unit="s", utc=True).tz_convert("Asia/Shanghai").isoformat()],
            "relative_sec": [0.0],
        }
    ).to_csv(directory / "events.csv", index=False)
    config = {
        "name": "test",
        "stages": [
            {
                "key": "test_stage",
                "name": "test stage",
                "duration": duration,
                "start_sec": 0.0,
                "end_sec": float(duration),
            }
        ],
    }
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if windows is not None:
        pd.DataFrame(
            windows,
            columns=["window_index", "window_start_sec", "window_end_sec", "stage", "marker", "record_time", "note"],
        ).to_csv(directory / "30_windows.csv", index=False)
