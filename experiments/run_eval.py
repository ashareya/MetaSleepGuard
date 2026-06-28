"""Evaluate a trained model or run a quick synthetic evaluation."""

from __future__ import annotations

import argparse

from MetaSleepGuard.experiments.common import load_public_records, output_dir, setup_logging, write_json
import numpy as np

from MetaSleepGuard.models.evaluate import evaluate_predictions, evaluate_probabilities
from MetaSleepGuard.models.train_xgb import (
    load_model_bundle,
    predict_class_probabilities,
    records_to_feature_dataset,
    train_subject_split,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset", default="sleep-edf", choices=["sleep-edf", "isruc"])
    parser.add_argument("--root", default=None)
    parser.add_argument("--task", default="5class", choices=["3class", "4class", "5class"])
    parser.add_argument("--max-subjects", type=int, default=4)
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--require-real-data", action="store_true")
    args = parser.parse_args()
    setup_logging()
    records = load_public_records(
        args.dataset,
        args.root,
        args.max_subjects,
        allow_synthetic=not args.require_real_data,
    )
    if args.model:
        bundle = load_model_bundle(args.model)
        data = records_to_feature_dataset(records, task=args.task, apply_preprocessing=not args.no_filter)
        if list(bundle.get("feature_names", [])) != data.feature_names:
            raise ValueError("model and evaluation data use different feature schemas")
        probabilities = predict_class_probabilities(bundle["model"], data.X, len(data.classes))
        y_pred = np.argmax(probabilities, axis=1)
        metrics = evaluate_predictions(data.y, y_pred, data.classes, data.subject_ids)
        metrics.update(evaluate_probabilities(data.y, probabilities))
        metrics["metadata"] = data.metadata
        metrics["model_training_metadata"] = bundle.get("metadata", {})
    else:
        metrics = train_subject_split(records, task=args.task, apply_preprocessing=not args.no_filter)["metrics"]
    metrics_path = write_json(metrics, output_dir() / "metrics" / f"{args.dataset}_{args.task}_eval.json")
    print(f"metrics={metrics_path}")
    synthetic = bool(any(record.metadata.get("synthetic") for record in records))
    print(f"synthetic_demo={synthetic}")
    print(f"accuracy={metrics['accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
