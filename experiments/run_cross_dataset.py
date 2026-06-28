"""Run Sleep-EDF -> ISRUC and ISRUC -> Sleep-EDF evaluation."""

from __future__ import annotations

import argparse

from MetaSleepGuard.experiments.common import load_public_records, output_dir, setup_logging, write_json
from MetaSleepGuard.models.cross_dataset_eval import bidirectional_cross_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleep-edf-root", default=None)
    parser.add_argument("--isruc-root", default=None)
    parser.add_argument("--task", default="5class", choices=["3class", "4class", "5class"])
    parser.add_argument("--max-subjects", type=int, default=4)
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--require-real-data", action="store_true")
    args = parser.parse_args()
    setup_logging()
    sleep_edf = load_public_records(
        "sleep-edf",
        args.sleep_edf_root,
        args.max_subjects,
        allow_synthetic=not args.require_real_data,
    )
    isruc = load_public_records(
        "isruc",
        args.isruc_root,
        args.max_subjects,
        allow_synthetic=not args.require_real_data,
    )
    metrics = bidirectional_cross_dataset(sleep_edf, isruc, task=args.task, apply_preprocessing=not args.no_filter)
    metrics_path = write_json(metrics, output_dir() / "metrics" / f"cross_dataset_{args.task}.json")
    print(f"cross_dataset_metrics={metrics_path}")
    print(f"synthetic_demo={any(record.metadata.get('synthetic') for record in sleep_edf + isruc)}")


if __name__ == "__main__":
    main()
