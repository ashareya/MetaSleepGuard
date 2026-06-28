import numpy as np

from MetaSleepGuard.quality.artifact_rules import detect_bad_channel, detect_bad_channel_names
from MetaSleepGuard.quality.quality_audit import _worst_grade, audit_epoch, audit_windows
from MetaSleepGuard.realtime.realtime_pipeline import RealtimePipeline


def test_quality_detects_flatline_and_dropout():
    sfreq = 250
    flat = np.zeros((2, sfreq * 30))
    result = audit_epoch(flat, sfreq, ["Ch2", "Ch7"])
    assert "flatline" in result.bad_flags
    assert "data_dropout" in result.bad_flags
    assert not result.is_reliable


def test_quality_windows_for_sine():
    sfreq = 250
    t = np.arange(sfreq * 60) / sfreq
    signals = np.vstack([20e-6 * np.sin(2 * np.pi * 10 * t), 15e-6 * np.sin(2 * np.pi * 6 * t)])
    results = audit_windows(signals, sfreq, ["Ch2", "Ch7"])
    assert len(results) == 2
    assert all(result.quality_grade in {"A", "B", "C", "D"} for result in results)


def test_realtime_quality_audits_raw_line_noise():
    sfreq = 250
    t = np.arange(sfreq * 30) / sfreq
    contaminated = np.vstack(
        [
            10e-6 * np.sin(2 * np.pi * 10 * t) + 50e-6 * np.sin(2 * np.pi * 50 * t),
            10e-6 * np.sin(2 * np.pi * 8 * t) + 50e-6 * np.sin(2 * np.pi * 50 * t),
        ]
    )
    row = RealtimePipeline().process_epoch(contaminated)
    assert "line_noise" in row["bad_flags"]


def test_two_channel_imbalance_and_worst_grade():
    epoch = np.vstack([np.ones(1000) * 1e-6, np.linspace(-100e-6, 100e-6, 1000)])
    assert detect_bad_channel(epoch)
    assert detect_bad_channel_names(epoch, ["Ch2", "Ch7"])
    assert _worst_grade(["A", "D", "B"]) == "D"


def test_six_required_artifact_classes():
    sfreq = 250
    t = np.arange(sfreq * 30) / sfreq
    clean = 10e-6 * np.sin(2 * np.pi * 10 * t)

    line = np.vstack([clean + 60e-6 * np.sin(2 * np.pi * 50 * t)] * 2)
    assert "line_noise" in audit_epoch(line, sfreq).bad_flags

    drift = np.vstack([clean + 80e-6 * np.sin(2 * np.pi * 0.2 * t)] * 2)
    assert "baseline_drift" in audit_epoch(drift, sfreq).bad_flags

    saturated = np.vstack([np.full(t.size, 600e-6), clean])
    assert "channel_saturation" in audit_epoch(saturated, sfreq).bad_flags

    flat = np.vstack([np.zeros(t.size), clean])
    assert "flatline" in audit_epoch(flat, sfreq).bad_flags

    motion = np.vstack([clean.copy(), clean.copy()])
    motion[:, sfreq * 10 : sfreq * 10 + 20] = 700e-6
    motion_result = audit_epoch(motion, sfreq)
    assert "motion_artifact" in motion_result.bad_flags
    assert not motion_result.is_reliable

    interrupted = np.vstack([clean.copy(), clean.copy()])
    interrupted[:, :sfreq] = np.nan
    assert "data_dropout" in audit_epoch(interrupted, sfreq).bad_flags
