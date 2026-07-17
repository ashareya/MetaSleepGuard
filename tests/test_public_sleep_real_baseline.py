from pathlib import Path
from unittest.mock import patch

import numpy as np

from MetaSleepGuard.models.public_sleep_real_baseline import (
    grouped_random_forest_baseline,
    map_labels,
    validate_metrics_schema,
)
from MetaSleepGuard.preprocessing.epoching import epoch_signal
from MetaSleepGuard.preprocessing.label_mapping import map_raw_stage
from MetaSleepGuard.reports.submission_pack import _manifest_rows
from MetaSleepGuard.reports.submission_pack import _real_items
from MetaSleepGuard.experiments.download_sleep_edf_subset import _byte_ranges
from MetaSleepGuard.experiments.download_isruc import (
    download_resumable,
    remote_size,
    select_nemar_manifest_rows,
    verify_nemar_checksum,
)
from MetaSleepGuard.datasets.public_sleep.loaders import find_isruc_records, parse_isruc_bids_events
from MetaSleepGuard.experiments.common import output_dir, project_root, repo_root
from MetaSleepGuard.experiments.run_public_sleep_real_baseline import _limitations, _readme
from MetaSleepGuard.models.train_xgb import build_classifier


def test_sleep_edf_raw_label_mapping():
    assert map_raw_stage("Sleep stage W") == "W"
    assert map_raw_stage("Sleep stage 1") == "N1"
    assert map_raw_stage("Sleep stage 2") == "N2"
    assert map_raw_stage("Sleep stage 3") == "N3"
    assert map_raw_stage("Sleep stage 4") == "N3"
    assert map_raw_stage("Sleep stage R") == "REM"
    assert map_raw_stage("Movement time") == "UNKNOWN"


def test_real_baseline_thirty_second_epoch_generation():
    signals = np.zeros((2, 100 * 65), dtype=float)
    epochs = epoch_signal(signals, sfreq=100, epoch_sec=30)
    assert epochs.shape == (2, 2, 3000)


def test_real_baseline_three_class_mapping():
    mapped = map_labels(["W", "N1", "N2", "N3", "REM"], "3class")
    assert mapped.tolist() == ["Wake", "NREM", "NREM", "NREM", "REM"]


def test_real_baseline_five_class_mapping():
    mapped = map_labels(["W", "N1", "N2", "N3", "REM"], "5class")
    assert mapped.tolist() == ["W", "N1", "N2", "N3", "REM"]


def _small_grouped_result():
    rng = np.random.default_rng(4)
    subjects = np.repeat([f"SC40{i}" for i in range(5)], 10)
    labels = np.tile(["W", "N1", "N2", "N3", "REM"], 10)
    features = rng.normal(size=(50, 8))
    return grouped_random_forest_baseline(features, labels, subjects, "5class")


def test_group_split_has_no_subject_leakage():
    metrics, _, splits = _small_grouped_result()
    assert metrics["split_method"].startswith("GroupKFold")
    assert metrics["subject_overlap_in_any_fold"] is False
    for split in splits:
        train = set(split["train_subject_ids"].split("|"))
        test = set(split["test_subject_ids"].split("|"))
        assert train.isdisjoint(test)
        assert split["subject_overlap_count"] == 0


def test_real_metrics_json_schema_is_complete():
    metrics, _, _ = _small_grouped_result()
    validate_metrics_schema(metrics)
    required = {
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "cohen_kappa",
        "per_class",
        "confusion_matrix",
        "class_distribution",
        "subject_ids",
    }
    assert required.issubset(metrics)


def _manifest_fixture(tmp_path: Path):
    repo = tmp_path / "repo"
    metrics_dir = repo / "outputs" / "metasleepguard_outputs" / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "synthetic.json").write_text(
        '{"metadata":{"synthetic_demo":true}}',
        encoding="utf-8",
    )
    real_raw = tmp_path / "real.edf"
    real_raw.write_bytes(b"edf")
    public_metric = tmp_path / "sleep_edf_5class_metrics.json"
    public_metric.write_text("{}", encoding="utf-8")
    rows = _manifest_rows(
        repo,
        tmp_path / "reports",
        [],
        [public_metric],
        {"formal_data_file": str(real_raw)},
        {"segment_files": [str(real_raw)]},
    )
    return rows, public_metric


def test_manifest_marks_real_public_metrics_for_accuracy(tmp_path):
    rows, public_metric = _manifest_fixture(tmp_path)
    row = next(item for item in rows if item["file_path"] == str(public_metric))
    assert row["data_type"] == "public_sleep_real_baseline"
    assert row["real_or_synthetic"] == "real_public_dataset"
    assert row["used_for_accuracy"] == "yes"
    assert row["used_for_quality"] == "no"
    assert row["note"] == "subject-level split, no epoch leakage"


def test_synthetic_metrics_never_become_accuracy_evidence(tmp_path):
    rows, _ = _manifest_fixture(tmp_path)
    synthetic = [row for row in rows if row["real_or_synthetic"] == "synthetic"]
    assert synthetic
    assert all(row["used_for_accuracy"] == "no" for row in synthetic)
    assert all("smoke test only" in row["note"] for row in synthetic)


def test_submission_text_drops_stale_public_data_status():
    public_summary = {
        "n_subjects": 5,
        "n_epochs_total": 5156,
    }
    text = _real_items({"coverage_ratio": 0.99, "sample_completeness_ratio": 0.99}, {"coverage_ratio": 0.97, "missing_duration_sec": 96}, public_summary)
    assert "公开睡眠数据尚未落地" not in text
    assert "真实 Sleep-EDF 专家标签基线" in text


def test_real_baseline_readme_reproduction_command_has_no_carriage_return():
    summary = {
        "n_subjects": 5,
        "n_epochs_total": 100,
        "split_method": "GroupKFold",
    }
    results = {
        "3class": {"accuracy": 0.8, "macro_f1": 0.7, "cohen_kappa": 0.6},
        "5class": {"accuracy": 0.7, "macro_f1": 0.6, "cohen_kappa": 0.5},
    }
    text = _readme(summary, results)
    assert ".\\run.ps1 -Task public-sleep-real-baseline -MaxSubjects 5 -Python $py" in text
    assert "\r" not in text


def test_real_baseline_limitations_use_actual_subject_count():
    text = _limitations({"n_subjects": 15})
    assert "15 名被试" in text
    assert "至少 5 名被试" not in text


def test_segmented_download_ranges_cover_file_once():
    ranges = _byte_ranges(101, 8)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 100
    assert sum(end - start + 1 for start, end in ranges) == 101
    assert all(left[1] + 1 == right[0] for left, right in zip(ranges, ranges[1:]))


def test_repository_and_output_roots_stay_inside_git_workspace():
    assert repo_root() == project_root()
    assert output_dir().is_relative_to(repo_root())


def test_decision_evidence_working_points_and_artifacts():
    from MetaSleepGuard.experiments.run_evidence_boost import run_artifact_injection, select_working_points

    curve = [
        {"threshold": 0.0, "coverage": 1.0, "risk": 0.3, "macro_f1": 0.6},
        {"threshold": 0.5, "coverage": 0.8, "risk": 0.2, "macro_f1": 0.7},
        {"threshold": 0.8, "coverage": 0.5, "risk": 0.1, "macro_f1": 0.8},
    ]
    points = select_working_points(curve)
    assert [row["mode"] for row in points] == ["high_coverage", "balanced", "high_trust"]
    artifacts = run_artifact_injection()
    assert len(artifacts["rows"]) == 7
    assert 0.0 <= artifacts["detection_recall"] <= 1.0


def test_isruc_remote_size_requires_valid_content_range():
    class Response:
        status_code = 206
        headers = {"Content-Range": "bytes 0-0/12345"}
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def raise_for_status(self): return None

    with patch("MetaSleepGuard.experiments.download_isruc.requests.get", return_value=Response()):
        assert remote_size("https://example.invalid/1.rar") == 12345


def test_isruc_resume_rejects_wrong_range_start(tmp_path):
    archive = tmp_path / "1.rar"
    archive.write_bytes(b"partial")

    class Response:
        status_code = 206
        headers = {"Content-Range": "bytes 0-9/20"}
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def raise_for_status(self): return None

    with patch("MetaSleepGuard.experiments.download_isruc.requests.get", return_value=Response()):
        try:
            download_resumable("https://example.invalid/1.rar", archive, expected_size=20)
        except RuntimeError as exc:
            assert "Unsafe resume response" in str(exc)
        else:
            raise AssertionError("unsafe Range response was accepted")
    assert archive.read_bytes() == b"partial"


def test_run_script_auto_selects_metabci_environments():
    script = (repo_root() / "run.ps1").read_text(encoding="utf-8")
    assert 'Find-CondaEnvironmentPython "metabci"' in script
    assert 'Find-CondaEnvironmentPython "metabci_stim"' in script
    assert '[string]$Python = "python"' not in script


def test_nemar_manifest_selects_requested_subgroup_subjects():
    def row(path, algorithm="git"):
        return {"path": path, "size": 1, "checksum_algorithm": algorithm, "checksum": "x", "bytes_url": "https://x"}

    rows = [
        row("README.md"),
        row("sub-I001/eeg/sub-I001_task-sleep_eeg.edf", "sha256"),
        row("sub-I001/eeg/sub-I001_task-sleep_events.tsv"),
        row("sub-I002/eeg/sub-I002_task-sleep_eeg.edf", "sha256"),
        row("sub-II001/eeg/sub-II001_task-sleep_eeg.edf", "sha256"),
    ]
    selected = select_nemar_manifest_rows(rows, subjects=1)
    paths = {item["path"] for item in selected}
    assert "README.md" in paths
    assert any(path.startswith("sub-I001/") for path in paths)
    assert not any(path.startswith("sub-I002/") or path.startswith("sub-II") for path in paths)


def test_nemar_sha256_and_git_blob_checksums(tmp_path):
    import hashlib

    path = tmp_path / "metadata.tsv"
    path.write_bytes(b"abc")
    verify_nemar_checksum(path, "sha256", hashlib.sha256(b"abc").hexdigest())
    git_hash = hashlib.sha1(b"blob 3\0abc").hexdigest()
    verify_nemar_checksum(path, "git", git_hash)


def test_nemar_bids_events_and_pair_discovery(tmp_path):
    eeg = tmp_path / "sub-I001" / "eeg" / "sub-I001_task-sleep_eeg.edf"
    eeg.parent.mkdir(parents=True)
    eeg.write_bytes(b"edf")
    events = eeg.with_name("sub-I001_task-sleep_events.tsv")
    events.write_text(
        "onset\tduration\ttrial_type\texpert_2\n"
        "0\t30\tSleep stage W\tSleep stage W\n"
        "30\t60\tSleep stage 2\tSleep stage 3\n"
        "90\t30\tSleep stage R\tSleep stage R\n",
        encoding="utf-8",
    )
    pairs = find_isruc_records(tmp_path)
    assert pairs == [(eeg, events)]
    first, second = parse_isruc_bids_events(events)
    assert first == ["W", "N2", "N2", "REM"]
    assert second == ["W", "N3", "N3", "REM"]


def test_required_xgboost_never_silently_falls_back():
    import sys

    with patch.dict(sys.modules, {"xgboost": None}):
        try:
            build_classifier(42, require_model="xgboost")
        except RuntimeError as exc:
            assert "required XGBoost" in str(exc)
        else:
            raise AssertionError("required XGBoost unexpectedly fell back")

    model, kind = build_classifier(42, require_model="random_forest")
    assert kind == "sklearn_random_forest_fallback"
    assert model.random_state == 42
