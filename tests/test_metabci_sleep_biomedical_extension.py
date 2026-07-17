import json
import subprocess
import sys

import mne
import numpy as np


def test_feature_extractor_causal_context_and_names():
    from metabci_sleep.algorithms import SleepFeatureExtractor

    sfreq = 20
    t = np.arange(600) / sfreq
    epochs = np.stack(
        [np.vstack([20 * np.sin(2 * np.pi * (i + 1) * t), 15 * np.cos(2 * np.pi * 2 * t)]) for i in range(3)]
    )
    extractor = SleepFeatureExtractor(sfreq, ["C1", "C2"], context_history=2)
    X = extractor.transform(epochs, subject_ids=["S1"] * 3)
    assert X.shape[0] == 3
    assert X.shape[1] == len(extractor.get_feature_names_out())
    base = X.shape[1] // 3
    assert np.allclose(X[0, : base * 2], 0)
    assert not np.allclose(X[2, :base], 0)


def test_feature_extractor_stream_history_persists_and_resets():
    from metabci_sleep.algorithms import SleepFeatureExtractor

    rng = np.random.default_rng(9)
    epochs = rng.normal(size=(2, 2, 120))
    extractor = SleepFeatureExtractor(4, ["C1", "C2"], context_history=1)
    first = extractor.transform_stream(epochs[:1], subject_ids=["S1"])
    second = extractor.transform_stream(epochs[1:], subject_ids=["S1"])
    base = first.shape[1] // 2
    assert np.allclose(first[0, :base], 0)
    assert not np.allclose(second[0, :base], 0)
    extractor.reset_stream("S1")
    reset = extractor.transform_stream(epochs[1:], subject_ids=["S1"])
    assert np.allclose(reset[0, :base], 0)


def test_sleep_staging_estimator_fit_predict_and_save(tmp_path):
    from sklearn.ensemble import RandomForestClassifier
    from metabci_sleep.algorithms import SleepStagingEstimator

    rng = np.random.default_rng(3)
    X = rng.normal(size=(18, 2, 120))
    y = np.asarray([0, 1, 2] * 6)
    estimator = SleepStagingEstimator(
        task="3class",
        sfreq=4,
        channel_names=["C1", "C2"],
        context_history=1,
        estimator=RandomForestClassifier(n_estimators=12, random_state=3),
    )
    estimator.fit(X, y, subject_ids=["S1"] * 9 + ["S2"] * 9)
    probabilities = estimator.predict_proba(X, subject_ids=["S1"] * 9 + ["S2"] * 9)
    assert probabilities.shape == (18, 3)
    assert np.allclose(probabilities.sum(axis=1), 1)
    path = estimator.save(tmp_path / "estimator.joblib")
    loaded = SleepStagingEstimator.load(path)
    assert loaded.predict(X[:2], subject_ids=["S1", "S1"]).shape == (2,)


def test_calibration_and_coverage_risk_wrappers():
    from sklearn.linear_model import LogisticRegression
    from metabci_sleep.algorithms import CoverageRiskEvaluator, ProbabilityCalibrator

    X = np.arange(24, dtype=float).reshape(12, 2)
    y = np.asarray([0, 1] * 6)
    base = LogisticRegression().fit(X, y)
    calibrator = ProbabilityCalibrator().fit(base, X, y)
    probabilities = calibrator.predict_proba(X)
    metrics = calibrator.evaluate(y, probabilities)
    curve = CoverageRiskEvaluator().curve(y, probabilities, thresholds=[0.0, 0.8])
    assert {"ece", "brier_score", "coverage_risk_curve"} <= set(metrics)
    assert len(curve) == 2 and curve[0]["coverage"] == 1.0


def test_sleep_metrics_handles_waso_rejection_and_score():
    from metabci_sleep.algorithms import SleepMetrics

    metrics = SleepMetrics().compute(["W", "N1", "N2", "W", "W", "N3", "REM", "暂不判定"])
    assert metrics["recording_minutes"] == 4.0
    assert metrics["total_sleep_minutes"] == 2.0
    assert metrics["sleep_onset_latency_minutes"] == 0.5
    assert metrics["waso_minutes"] == 1.0
    assert metrics["awakenings_after_sleep_onset"] == 1
    assert metrics["unknown_or_rejected_epochs"] == 1
    assert 0 <= metrics["engineering_sleep_score"] <= 100
    assert "not a clinical" in metrics["score_disclaimer"]


def test_window_integrity_and_worker_layouts():
    from metabci_sleep.realtime import OpenBCISleepWorker, WindowIntegrityAuditor

    sfreq = 10
    timestamps = np.arange(300) / sfreq
    signals_uv = np.vstack([np.sin(timestamps), np.cos(timestamps)]) * 20
    rows = WindowIntegrityAuditor(sfreq=sfreq).audit(timestamps, signals_uv)
    assert len(rows) == 1
    assert rows[0]["sample_count"] == 300
    assert rows[0]["coverage_ratio"] == 1.0
    assert rows[0]["quality_grade"] == "A"

    worker = OpenBCISleepWorker(sfreq=sfreq, channel_names=["C1", "C2"])
    worker.pre()
    outputs = worker.consume(signals_uv.T)
    assert len(outputs) == 1
    assert outputs[0]["reason"] == "no_model_loaded"
    worker.post()

    incomplete = WindowIntegrityAuditor(sfreq=sfreq).audit(timestamps[:20], signals_uv[:, :20])
    assert incomplete[0]["quality_grade"] == "D"
    assert incomplete[0]["trusted_output"] == "暂不判定"


def test_worker_import_does_not_leave_log_file(tmp_path):
    code = (
        "from metabci_sleep.realtime import OpenBCISleepWorker; "
        "OpenBCISleepWorker(sfreq=10, channel_names=['C1','C2'])"
    )
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)
    assert not (tmp_path / "log.txt").exists()


def test_report_builder_and_brainstim_protocol(tmp_path):
    from metabci_sleep.brainstim import SleepCalibrationProtocol
    from metabci_sleep.reporting import SleepReportBuilder

    rows = [
        {"window_start_time": 0, "window_end_time": 30, "stage": "W", "confidence": 0.9, "accepted": True, "quality_grade": "A", "bad_flags": "", "reason": "accepted"},
        {"window_start_time": 30, "window_end_time": 60, "stage": "N2", "confidence": 0.8, "accepted": True, "quality_grade": "A", "bad_flags": "", "reason": "accepted"},
    ]
    result = SleepReportBuilder(tmp_path / "report").build(rows, {"source": "synthetic"})
    assert all((tmp_path / "report" / name).exists() for name in ["report.html", "report.md", "windows.csv", "summary.json"])
    summary = json.loads((tmp_path / "report" / "summary.json").read_text(encoding="utf-8"))
    assert "engineering_sleep_score" in summary["sleep_metrics"]

    protocol = SleepCalibrationProtocol()
    marker_path = protocol.run(tmp_path / "markers.csv", dry_run=True, use_psychopy=False, countdown_sec=0)
    assert marker_path.exists()
    assert len(protocol.event_plan()) >= 5


def test_isruc_dataset_standard_structure_with_mocked_files(tmp_path):
    from unittest.mock import patch
    from metabci_sleep.datasets import ISRUCSleep

    subject_dir = tmp_path / "subject1"
    subject_dir.mkdir()
    edf = subject_dir / "record.edf"
    labels = subject_dir / "record.txt"
    edf.touch()
    labels.write_text("0\n1\n2\n3\n5\n", encoding="utf-8")
    raw = mne.io.RawArray(
        np.zeros((2, 1500)),
        mne.create_info(["C3", "C4"], 10, ["eeg", "eeg"]),
        verbose="ERROR",
    )
    with (
        patch("metabci_sleep.datasets.isruc.find_isruc_records", return_value=[(edf, labels)]),
        patch("metabci_sleep.datasets.isruc.mne.io.read_raw_edf", side_effect=lambda *a, **k: raw.copy()),
    ):
        dataset = ISRUCSleep(tmp_path)
        nested = dataset.get_data(dataset.subjects)
    returned = nested["subject1"]["session_1"]["run_1"]
    assert returned.annotations.description.tolist() == ["1", "2", "3", "4", "5"]


def test_sleep_staging_four_class_mapping():
    from MetaSleepGuard.tests.test_metabci_sleep_extension import _synthetic_dataset
    from metabci_sleep.paradigms import SleepStaging

    _, y, meta = SleepStaging("4class").get_data(_synthetic_dataset())
    assert y.tolist() == [0, 1, 1, 2, 3]
    assert meta["stage"].tolist() == ["W", "LIGHT", "LIGHT", "N3", "REM"]


def test_official_isruc_rec_and_two_scorers_are_discovered(tmp_path):
    from MetaSleepGuard.datasets.public_sleep.loaders import find_isruc_records, parse_isruc_labels

    subject = tmp_path / "SubgroupI" / "1"
    subject.mkdir(parents=True)
    recording = subject / "1.rec"
    recording.write_bytes(b"EDF")
    scorer1 = subject / "1_1.txt"
    scorer2 = subject / "1_2.txt"
    scorer1.write_text("Epoch Stage\n0 0\n1 2\n2 5\n3 9\n", encoding="utf-8")
    scorer2.write_text("0,0\n1,1\n2,5\n", encoding="utf-8")

    assert find_isruc_records(tmp_path, scorer=1) == [(recording, scorer1)]
    assert find_isruc_records(tmp_path, scorer=2) == [(recording, scorer2)]
    assert parse_isruc_labels(scorer1) == ["W", "N2", "REM", "UNKNOWN"]
