from pathlib import Path

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
from MetaSleepGuard.experiments.run_public_sleep_real_baseline import _readme


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
    metrics_dir = repo / "_codex_tmp" / "metasleepguard_outputs" / "metrics"
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
    assert ".\\run.ps1 -Task public-sleep-real-baseline -Python $py" in text
    assert "\r" not in text
