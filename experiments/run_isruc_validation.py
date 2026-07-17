"""Run real NEMAR ISRUC internal and bidirectional cross-dataset validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupKFold

from MetaSleepGuard.experiments.common import create_run_dir, load_public_records, repo_root, write_json
from MetaSleepGuard.models.cross_dataset_eval import bidirectional_cross_dataset
from MetaSleepGuard.models.evaluate import evaluate_predictions, evaluate_probabilities
from MetaSleepGuard.models.train_xgb import (
    build_classifier,
    fit_with_subject_calibration,
    predict_class_probabilities,
    records_to_feature_dataset,
)


PROVENANCE = {
    "dataset": "ISRUC-Sleep",
    "distribution": "NEMAR nm000111 v1.0.1",
    "dataset_id": "nm000111",
    "version": "v1.0.1",
    "nemar_doi": "10.82901/nemar.nm000111",
    "isruc_paper_doi": "10.1016/j.cmpb.2015.10.013",
    "subgroup": "I",
    "scorer": 1,
}


def grouped_validation(records, task: str, random_state: int = 42, require_model: str | None = None) -> dict:
    dataset = records_to_feature_dataset(records, task=task)
    subjects = np.asarray(dataset.subject_ids, dtype=str)
    unique_subjects = np.unique(subjects)
    if unique_subjects.size < 5:
        raise ValueError("ISRUC formal validation requires at least five real subjects")
    splitter = GroupKFold(n_splits=5)
    predictions = np.zeros(dataset.y.size, dtype=int)
    probabilities = np.zeros((dataset.y.size, len(dataset.classes)), dtype=float)
    folds = []
    model_kinds = []
    model_configurations = []
    for fold, (train_index, test_index) in enumerate(
        splitter.split(dataset.X, dataset.y, groups=subjects), start=1
    ):
        train_subjects = sorted(set(subjects[train_index]))
        test_subjects = sorted(set(subjects[test_index]))
        overlap = sorted(set(train_subjects) & set(test_subjects))
        if overlap:
            raise RuntimeError(f"subject leakage in fold {fold}: {overlap}")
        fold_seed = random_state + fold
        model, model_kind = build_classifier(fold_seed, require_model=require_model)
        configuration = {
            "fold": fold,
            "model_kind": model_kind,
            "random_state": fold_seed,
            "parameters": model.get_params(deep=False),
        }
        if model_kind == "xgboost":
            import xgboost

            configuration["xgboost_version"] = xgboost.__version__
        model_configurations.append(configuration)
        train_mask = np.zeros(dataset.y.size, dtype=bool)
        train_mask[train_index] = True
        model, calibration = fit_with_subject_calibration(
            model, dataset.X, dataset.y, subjects, train_mask, random_state + fold
        )
        fold_probabilities = predict_class_probabilities(model, dataset.X[test_index], len(dataset.classes))
        probabilities[test_index] = fold_probabilities
        predictions[test_index] = np.argmax(fold_probabilities, axis=1)
        model_kinds.append(model_kind)
        folds.append(
            {
                "fold": fold,
                "train_subject_ids": train_subjects,
                "test_subject_ids": test_subjects,
                "subject_overlap_count": 0,
                "n_train_epochs": int(train_index.size),
                "n_test_epochs": int(test_index.size),
                "calibration": calibration,
            }
        )
    metrics = evaluate_predictions(dataset.y, predictions, dataset.classes, subjects)
    metrics.update(evaluate_probabilities(dataset.y, probabilities))
    metrics.update(
        {
            **PROVENANCE,
            "task": task,
            "real_public_dataset": True,
            "synthetic_demo": False,
            "n_subjects": int(unique_subjects.size),
            "subject_ids": unique_subjects.tolist(),
            "n_epochs_total": int(dataset.y.size),
            "classes": dataset.classes,
            "split_method": "GroupKFold(n_splits=5, group=subject_id)",
            "subject_overlap_in_any_fold": False,
            "model_kinds": sorted(set(model_kinds)),
            "model_configurations": model_configurations,
            "folds": folds,
        }
    )
    return metrics


def data_manifest(records) -> dict:
    rows = []
    for record in records:
        valid = [label for label in record.labels[: record.n_epochs_from_signal] if label != "UNKNOWN"]
        rows.append(
            {
                "subject_id": record.subject_id,
                "sampling_rate_hz": record.sfreq,
                "channels": record.channel_names,
                "signal_epochs": record.n_epochs_from_signal,
                "label_epochs": len(record.labels),
                "valid_epochs": len(valid),
                "class_distribution": {stage: valid.count(stage) for stage in ("W", "N1", "N2", "N3", "REM")},
                "metadata": _portable_metadata(record.metadata),
            }
        )
    return {**PROVENANCE, "n_subjects": len(rows), "subjects": rows}


def _portable_metadata(value, key: str = ""):
    """Remove machine-specific directory prefixes from exported manifests."""

    if isinstance(value, dict):
        return {name: _portable_metadata(item, name) for name, item in value.items()}
    if isinstance(value, list):
        return [_portable_metadata(item, key) for item in value]
    if isinstance(value, str) and ("path" in key.lower() or "file" in key.lower()):
        return Path(value).name
    return value


def write_metric_artifacts(run_dir: Path, name: str, metrics: dict) -> None:
    write_json(metrics, run_dir / f"{name}.json")
    per_class = metrics.get("per_class", {})
    if per_class:
        with (run_dir / f"{name}_per_class.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["class", "precision", "recall", "f1-score", "support"])
            writer.writeheader()
            for label, values in per_class.items():
                writer.writerow({"class": label, **{key: values.get(key) for key in writer.fieldnames[1:]}})
    matrix = np.asarray(metrics.get("confusion_matrix", []), dtype=int)
    if matrix.size:
        labels = metrics.get("classes") or list(per_class)
        figure, axis = plt.subplots(figsize=(6, 5))
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_xticks(range(len(labels)), labels=labels, rotation=30)
        axis.set_yticks(range(len(labels)), labels=labels)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title(name)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(run_dir / f"{name}_confusion_matrix.png", dpi=180)
        plt.close(figure)


def execute_validation(isruc, sleep_edf, subjects: int, run_dir: Path, require_model: str | None = None) -> dict:
    """Run both tasks and write artifacts; callers own status handling."""

    write_json(data_manifest(isruc), run_dir / "data_manifest.json")
    summary = {**PROVENANCE, "n_subjects": subjects, "required_model": require_model, "tasks": {}}
    for task in ("3class", "5class"):
        internal = grouped_validation(isruc, task, require_model=require_model)
        cross = bidirectional_cross_dataset(sleep_edf, isruc, task=task, require_model=require_model)
        cross["provenance"] = PROVENANCE
        write_metric_artifacts(run_dir, f"isruc_internal_{task}", internal)
        for direction, metrics in cross.items():
            if direction != "provenance":
                metrics.update(PROVENANCE)
                write_metric_artifacts(run_dir, f"cross_{direction}_{task}", metrics)
        write_json(cross, run_dir / f"cross_dataset_{task}.json")
        summary["tasks"][task] = {
            "internal": {key: internal[key] for key in ("accuracy", "macro_f1", "weighted_f1", "cohen_kappa")},
            "cross": {
                direction: {key: values[key] for key in ("accuracy", "macro_f1", "weighted_f1", "cohen_kappa")}
                for direction, values in cross.items()
                if direction != "provenance"
            },
        }
    write_json(summary, run_dir / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isruc-root", default=str(repo_root() / "data/public_sleep/isruc_raw/nemar_v1_0_1"))
    parser.add_argument("--sleep-edf-root", default=str(repo_root() / "data/public_sleep/sleep_edf_raw"))
    parser.add_argument("--subjects", type=int, default=15)
    parser.add_argument("--require-model", choices=["xgboost", "random_forest"], default=None)
    args = parser.parse_args()
    if args.subjects < 5:
        raise SystemExit("formal ISRUC validation requires at least five subjects")
    run_dir = create_run_dir("isruc_validation")
    status_path = run_dir / "status.json"
    write_json({**PROVENANCE, "status": "running", "requested_subjects": args.subjects}, status_path)
    try:
        isruc = load_public_records("isruc", args.isruc_root, args.subjects, allow_synthetic=False)
        if len({record.subject_id for record in isruc}) < args.subjects:
            raise RuntimeError("requested ISRUC subject count was not fully loaded")
        sleep_edf = load_public_records("sleep-edf", args.sleep_edf_root, args.subjects, allow_synthetic=False)
    except Exception as exc:
        write_json(
            {
                **PROVENANCE,
                "status": "failed",
                "requested_subjects": args.subjects,
                "error": f"{type(exc).__name__}: {exc}",
                "metrics_generated": False,
            },
            status_path,
        )
        print(f"isruc_validation_status={status_path}")
        raise
    try:
        summary = execute_validation(isruc, sleep_edf, args.subjects, run_dir, require_model=args.require_model)
    except Exception as exc:
        write_json(
            {
                **PROVENANCE,
                "status": "failed",
                "requested_subjects": args.subjects,
                "error": f"{type(exc).__name__}: {exc}",
                "metrics_generated": False,
            },
            status_path,
        )
        print(f"isruc_validation_status={status_path}")
        raise
    write_json(
        {**PROVENANCE, "status": "complete", "requested_subjects": args.subjects, "metrics_generated": True},
        status_path,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"isruc_validation_dir={run_dir}")


if __name__ == "__main__":
    main()
