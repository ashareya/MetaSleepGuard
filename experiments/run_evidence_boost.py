"""Generate leakage-safe ablation, calibration, rejection, and artifact evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import GroupKFold

from MetaSleepGuard.experiments.common import create_run_dir, repo_root
from MetaSleepGuard.experiments.run_public_sleep_real_baseline import (
    ensure_sleep_edf_data,
    load_feature_dataset,
)
from MetaSleepGuard.features.causal_context import append_causal_context
from MetaSleepGuard.quality.quality_audit import audit_epoch
from MetaSleepGuard.rejection.calibration import (
    calibrate_classifier,
    expected_calibration_error,
    multiclass_brier_score,
)
from MetaSleepGuard.rejection.coverage_risk import coverage_risk_curve


CLASSES = ["W", "N1", "N2", "N3", "REM"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(repo_root() / "data/public_sleep/sleep_edf_raw"))
    parser.add_argument("--subjects", type=int, default=15)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    pairs = ensure_sleep_edf_data(Path(args.data_root), list(range(args.subjects)), 1, allow_download=False)
    dataset = load_feature_dataset(pairs[: args.subjects])
    output = Path(args.output_root).resolve() if args.output_root else create_run_dir("decision_evidence")
    output.mkdir(parents=True, exist_ok=True)

    base = np.asarray(dataset["features"], dtype=float)
    labels = np.asarray(dataset["labels"], dtype=str)
    subjects = np.asarray(dataset["subject_ids"], dtype=str)
    names = list(dataset["feature_names"])
    single_indices = [index for index, name in enumerate(names) if name.startswith("fpz_cz_")]
    configurations = {
        "single_epoch_dual_channel": base,
        "causal_context_dual_channel": append_causal_context(base, subjects, history=2),
        "single_epoch_single_channel": base[:, single_indices],
    }
    split_signature = _split_signature(subjects)
    summaries = []
    detailed: dict[str, dict] = {}
    for name, features in configurations.items():
        result = grouped_probability_evaluation(features, labels, subjects, calibrate=False)
        result["configuration"] = name
        result["feature_count"] = int(features.shape[1])
        result["split_signature"] = split_signature
        summaries.append(_summary_row(result))
        detailed[name] = result

    calibrated = grouped_probability_evaluation(configurations["causal_context_dual_channel"], labels, subjects, calibrate=True)
    calibrated["configuration"] = "causal_context_dual_channel_calibrated"
    calibrated["feature_count"] = int(configurations["causal_context_dual_channel"].shape[1])
    calibrated["split_signature"] = split_signature
    summaries.append(_summary_row(calibrated))
    detailed[calibrated["configuration"]] = calibrated

    curve = calibrated["coverage_risk_curve"]
    working_points = select_working_points(curve)
    artifacts = run_artifact_injection()
    payload = {
        "real_public_dataset": True,
        "synthetic_staging_metrics": False,
        "dataset": "Sleep-EDF Expanded",
        "n_subjects": int(np.unique(subjects).size),
        "n_epochs": int(labels.size),
        "split_method": "GroupKFold(n_splits=5, group=subject_id)",
        "subject_overlap_in_any_fold": False,
        "random_state": 42,
        "configurations": detailed,
        "working_points": working_points,
        "artifact_injection": artifacts,
        "evidence_boundary": (
            "Artifact injections validate deterministic quality rules only; "
            "they are not sleep-staging accuracy evidence."
        ),
    }
    _write_json(output / "summary.json", payload)
    _write_csv(output / "ablation_comparison.csv", summaries)
    _write_csv(output / "coverage_risk_curve.csv", curve)
    _write_csv(output / "artifact_injection_results.csv", artifacts["rows"])
    _plot_ablation(summaries, output / "ablation_comparison.png")
    _plot_coverage_risk(curve, working_points, output / "coverage_risk_curve.png")
    (output / "CURRENT_CONCLUSIONS_AND_BOUNDARIES.md").write_text(
        _conclusions(summaries, working_points, artifacts), encoding="utf-8"
    )
    print(f"evidence_output={output}")
    print(f"ablation_rows={len(summaries)}")
    print(f"artifact_cases={len(artifacts['rows'])}")


def grouped_probability_evaluation(
    features: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    calibrate: bool,
    random_state: int = 42,
) -> dict:
    encoded = np.asarray([CLASSES.index(label) for label in labels], dtype=int)
    probabilities = np.zeros((encoded.size, len(CLASSES)), dtype=float)
    folds = []
    splitter = GroupKFold(n_splits=min(5, np.unique(subjects).size))
    for fold, (train, test) in enumerate(splitter.split(features, encoded, groups=subjects), start=1):
        train_subjects = sorted(set(subjects[train]))
        test_subjects = sorted(set(subjects[test]))
        if set(train_subjects) & set(test_subjects):
            raise RuntimeError("subject leakage detected")
        model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=random_state + fold,
            n_jobs=-1,
            min_samples_leaf=2,
        )
        calibration_subjects: list[str] = []
        if calibrate and len(train_subjects) >= 4:
            calibration_subjects = train_subjects[-max(1, len(train_subjects) // 4) :]
            calibration_mask = np.isin(subjects[train], calibration_subjects)
            fit = train[~calibration_mask]
            calibration = train[calibration_mask]
            model.fit(features[fit], encoded[fit])
            if set(encoded[calibration]) == set(range(len(CLASSES))):
                model = calibrate_classifier(model, features[calibration], encoded[calibration])
            else:
                model.fit(features[train], encoded[train])
                calibration_subjects = []
        else:
            model.fit(features[train], encoded[train])
        raw = np.asarray(model.predict_proba(features[test]), dtype=float)
        aligned = np.zeros((len(test), len(CLASSES)), dtype=float)
        aligned[:, np.asarray(model.classes_, dtype=int)] = raw
        probabilities[test] = aligned
        folds.append(
            {
                "fold": fold,
                "train_subjects": train_subjects,
                "calibration_subjects": calibration_subjects,
                "test_subjects": test_subjects,
                "subject_overlap_count": 0,
            }
        )
    predicted = np.argmax(probabilities, axis=1)
    return {
        "accuracy": float(accuracy_score(encoded, predicted)),
        "macro_f1": float(f1_score(encoded, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(encoded, predicted, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(encoded, predicted)),
        "ece": expected_calibration_error(encoded, probabilities),
        "brier_score": multiclass_brier_score(encoded, probabilities),
        "calibrated": bool(calibrate),
        "folds": folds,
        "coverage_risk_curve": coverage_risk_curve(encoded, probabilities, thresholds=np.linspace(0, 0.95, 20)),
    }


def select_working_points(curve: list[dict]) -> list[dict]:
    targets = [("high_coverage", 0.90), ("balanced", 0.75), ("high_trust", 0.50)]
    output = []
    for name, target in targets:
        eligible = [row for row in curve if row["coverage"] >= target]
        row = min(eligible, key=lambda item: (item["risk"], abs(item["coverage"] - target))) if eligible else curve[0]
        output.append({"mode": name, **row})
    return output


def run_artifact_injection(sfreq: float = 250.0) -> dict:
    rng = np.random.default_rng(42)
    time = np.arange(int(sfreq * 30)) / sfreq
    clean = np.vstack(
        [
            25e-6 * np.sin(2 * np.pi * 10 * time) + rng.normal(0, 3e-6, time.size),
            22e-6 * np.sin(2 * np.pi * 8 * time) + rng.normal(0, 3e-6, time.size),
        ]
    )
    cases = {
        "clean": clean,
        "flatline": np.zeros_like(clean),
        "dropout": np.where(np.arange(clean.shape[1]) < clean.shape[1] * 0.3, 0.0, clean),
        "line_noise_50hz": clean + 80e-6 * np.sin(2 * np.pi * 50 * time),
        "baseline_drift": clean + 180e-6 * np.sin(2 * np.pi * 0.15 * time),
        "amplitude_spike": clean.copy(),
        "high_frequency_noise": clean + rng.normal(0, 180e-6, clean.shape),
    }
    cases["amplitude_spike"][:, int(sfreq * 10) : int(sfreq * 10) + 10] = 2e-3
    expected = {
        "clean": set(),
        "flatline": {"flatline"},
        "dropout": {"data_dropout"},
        "line_noise_50hz": {"line_noise"},
        "baseline_drift": {"baseline_drift"},
        "amplitude_spike": {"abnormal_amplitude", "motion_artifact"},
        "high_frequency_noise": {"abnormal_amplitude", "motion_artifact"},
    }
    rows = []
    true_positive = false_negative = false_positive = 0
    for name, signal in cases.items():
        result = audit_epoch(signal, sfreq, ["EEG1", "EEG2"])
        detected = set(result.bad_flags)
        wanted = expected[name]
        hit = bool(detected & wanted) if wanted else not detected
        true_positive += int(bool(wanted) and bool(detected & wanted))
        false_negative += int(bool(wanted) and not bool(detected & wanted))
        false_positive += int(not wanted and bool(detected))
        rows.append(
            {
                "case": name,
                "expected_flags": "|".join(sorted(wanted)),
                "detected_flags": "|".join(sorted(detected)),
                "quality_grade": result.quality_grade,
                "is_reliable": result.is_reliable,
                "expected_detected": hit,
            }
        )
    injected = len(cases) - 1
    return {
        "rows": rows,
        "detection_recall": true_positive / injected,
        "clean_false_positive_rate": float(false_positive),
        "false_negative_cases": false_negative,
    }


def _split_signature(subjects: np.ndarray) -> list[dict]:
    rows = []
    dummy = np.zeros(len(subjects))
    for fold, (train, test) in enumerate(GroupKFold(5).split(dummy, dummy, groups=subjects), start=1):
        rows.append(
            {
                "fold": fold,
                "train_subjects": sorted(set(subjects[train])),
                "test_subjects": sorted(set(subjects[test])),
            }
        )
    return rows


def _summary_row(result: dict) -> dict:
    return {
        key: result[key]
        for key in (
            "configuration",
            "feature_count",
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "cohen_kappa",
            "ece",
            "brier_score",
            "calibrated",
        )
    }


def _plot_ablation(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["configuration"].replace("_", "\n") for row in rows]
    x = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(x - 0.18, [row["accuracy"] for row in rows], 0.36, label="Accuracy")
    axis.bar(x + 0.18, [row["macro_f1"] for row in rows], 0.36, label="Macro-F1")
    axis.set_ylim(0, 1)
    axis.set_xticks(x, labels)
    axis.set_title("Sleep-EDF ablation with identical subject folds")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_coverage_risk(curve: list[dict], points: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot([row["coverage"] for row in curve], [row["risk"] for row in curve], marker="o")
    for point in points:
        axis.scatter(point["coverage"], point["risk"], s=60, label=point["mode"])
    axis.set_xlabel("Coverage")
    axis.set_ylabel("Risk among accepted epochs")
    axis.set_title("Trusted rejection: coverage-risk")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _conclusions(rows: list[dict], points: list[dict], artifacts: dict) -> str:
    best = max(rows, key=lambda row: row["macro_f1"])
    point_lines = "\n".join(
        f"- {row['mode']}: threshold={row['threshold']:.2f}, coverage={row['coverage']:.3f}, risk={row['risk']:.3f}"
        for row in points
    )
    return f"""# 当前结论与证据边界

- 本页消融结果来自15名Sleep-EDF真实被试,并使用完全相同的GroupKFold被试划分。
- 当前Macro-F1最高配置为`{best['configuration']}`,Macro-F1={best['macro_f1']:.4f}。
- 概率校准使用训练折内部留出的被试,不接触测试被试。
- OpenBCI数据不参与本页睡眠分期准确率计算。

## 可信拒识工作点

{point_lines}

## 可控伪迹

- 伪迹检测召回率:{artifacts['detection_recall']:.3f}
- 干净样本误报计数:{artifacts['clean_false_positive_rate']:.0f}
- 伪迹注入只验证质量规则,不作为睡眠分期准确率证据。

## ISRUC

只有真实ISRUC文件完成加载和复测后,才加入ISRUC内部与跨数据集指标。
"""


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
