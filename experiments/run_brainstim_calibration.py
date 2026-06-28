"""Run Brainstim/PsychoPy calibration paradigm."""

from __future__ import annotations

import argparse

from MetaSleepGuard.brainstim_task.calibration_task import run_calibration_task
from MetaSleepGuard.experiments.common import create_run_dir, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-psychopy", action="store_true")
    args = parser.parse_args()
    setup_logging()
    out = args.output_csv or (create_run_dir("brainstim") / "calibration_markers.csv")
    path = run_calibration_task(out, dry_run=args.dry_run, use_psychopy=not args.no_psychopy)
    print(f"marker_log={path}")


if __name__ == "__main__":
    main()
