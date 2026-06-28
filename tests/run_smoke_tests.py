"""No-pytest smoke tests for minimal environments.

Run from the repository root:

    python -m MetaSleepGuard.tests.run_smoke_tests
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from MetaSleepGuard.datasets.public_sleep import generate_synthetic_public_records
from MetaSleepGuard.features import extract_features_from_epochs
from MetaSleepGuard.models.train_xgb import train_subject_split
from MetaSleepGuard.models.sleep_inference import SleepInference
from MetaSleepGuard.preprocessing.epoching import epoch_signal
from MetaSleepGuard.quality.quality_audit import audit_epoch
from MetaSleepGuard.realtime.openbci_file_loader import load_openbci_csv
from MetaSleepGuard.realtime.openbci_brainflow_stream import SyntheticBrainFlowStream
from MetaSleepGuard.realtime.realtime_pipeline import RealtimePipeline
from MetaSleepGuard.rejection.rejector import ActiveRejector
from MetaSleepGuard.reports.report_generator import generate_html_report


def main() -> None:
    sfreq = 250
    t = np.arange(sfreq * 60) / sfreq
    signals = np.vstack([20e-6 * np.sin(2 * np.pi * 10 * t), 15e-6 * np.sin(2 * np.pi * 6 * t)])
    epochs = epoch_signal(signals, sfreq)
    assert epochs.shape == (2, 2, sfreq * 30)
    X, names = extract_features_from_epochs(epochs, sfreq, ["Ch2", "Ch7"])
    assert X.shape[0] == 2 and X.shape[1] == len(names)
    flat_quality = audit_epoch(np.zeros((2, sfreq * 30)), sfreq, ["Ch2", "Ch7"])
    assert "flatline" in flat_quality.bad_flags
    decision = ActiveRejector().decide([0.9, 0.1], ["W", "N1"], "A", True)
    assert decision.accepted
    stream = SyntheticBrainFlowStream()
    stream.start()
    chunk = stream.read(seconds=31)
    assert chunk.data.shape == (2, sfreq * 31)
    rows = RealtimePipeline(sfreq=chunk.sfreq, channel_names=chunk.channel_names).append_and_process(chunk.data)
    assert len(rows) == 1
    records = generate_synthetic_public_records(n_subjects=4, n_epochs=6)
    result = train_subject_split(records, task="5class", apply_preprocessing=False)
    assert "macro_f1" in result["metrics"]
    assert "ece" in result["metrics"] and "brier_score" in result["metrics"]
    assert "coverage_risk_curve" in result["metrics"] and "rejection" in result["metrics"]
    inference = SleepInference(result["bundle"])
    online = inference.predict_epoch(epochs[0], sfreq, ["Ch2", "Ch7"])
    assert np.isclose(sum(online["probabilities"].values()), 1.0)
    assert np.any(inference._base_vector(epochs[0], sfreq, ["Ch2", "Ch7"]) != 0.0)
    inference.reset()
    bad_epoch = np.zeros_like(epochs[0])
    bad_online = inference.predict_epoch(bad_epoch, sfreq, ["Ch2", "Ch7"])
    assert not bad_online["accepted"] and np.allclose(inference.history[-1], 0.0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        csv_path = tmp_path / "openbci.csv"
        pd.DataFrame(
            {
                "Sample Index": np.arange(sfreq),
                "EXG Channel 0": np.sin(2 * np.pi * 10 * np.arange(sfreq) / sfreq),
                "EXG Channel 1": np.sin(2 * np.pi * 6 * np.arange(sfreq) / sfreq),
                "EXG Channel 6": np.sin(2 * np.pi * 3 * np.arange(sfreq) / sfreq),
            }
        ).to_csv(csv_path, index=False)
        replay = load_openbci_csv(csv_path, channel_indices=(1, 2), sfreq=sfreq)
        assert replay.signals.shape == (2, sfreq)
        assert np.max(np.abs(replay.signals)) <= 1.01e-6
        report = generate_html_report(
            tmp_path / "report.html",
            {"mode": "smoke", "device": "synthetic"},
            rows,
        )
        assert report.exists()
    print("MetaSleepGuard smoke tests passed")


if __name__ == "__main__":
    main()
