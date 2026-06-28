"""Train XGBoost/fallback baseline on public sleep records."""

from __future__ import annotations

import argparse

from MetaSleepGuard.experiments.common import load_public_records, output_dir, setup_logging, write_json
from MetaSleepGuard.models.train_xgb import train_subject_split


def main() -> None:
    parser = argparse.ArgumentParser()
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
    out_dir = output_dir()
    model_path = out_dir / "models" / f"{args.dataset}_{args.task}_baseline.joblib"
    result = train_subject_split(records, task=args.task, output_path=model_path, apply_preprocessing=not args.no_filter)
    metrics_path = write_json(result["metrics"], out_dir / "metrics" / f"{args.dataset}_{args.task}_metrics.json")
    print(f"model={model_path}")
    print(f"metrics={metrics_path}")
    print(f"synthetic_demo={result['metrics']['metadata']['synthetic_demo']}")
    print(f"macro_f1={result['metrics']['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
