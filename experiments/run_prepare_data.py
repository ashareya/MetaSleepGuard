"""Prepare or smoke-test public sleep data."""

from __future__ import annotations

import argparse
import joblib

from MetaSleepGuard.experiments.common import load_public_records, output_dir, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="sleep-edf", choices=["sleep-edf", "isruc"])
    parser.add_argument("--root", default=None)
    parser.add_argument("--max-subjects", type=int, default=4)
    args = parser.parse_args()
    setup_logging()
    records = load_public_records(args.dataset, args.root, args.max_subjects)
    out = output_dir() / f"{args.dataset}_records.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(records, out)
    synthetic = any(record.metadata.get("synthetic") for record in records)
    print(f"Saved {len(records)} records to {out}")
    print(f"synthetic_demo_records={synthetic}")


if __name__ == "__main__":
    main()

