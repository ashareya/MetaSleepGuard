"""Audit Boruikang BDF/FIF files."""

from __future__ import annotations

import argparse

from MetaSleepGuard.experiments.common import create_run_dir, setup_logging
from MetaSleepGuard.quality.quality_audit import audit_bdf_fif_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    setup_logging()
    out_dir = args.output_dir or create_run_dir("bdf_fif_audit")
    csv_path = audit_bdf_fif_directory(args.input_dir, out_dir)
    print(f"audit_csv={csv_path}")


if __name__ == "__main__":
    main()
