"""Generate metric, documentation, and demo-video submission materials."""

from __future__ import annotations

import argparse

from MetaSleepGuard.experiments.common import repo_root
from MetaSleepGuard.reports.submission_pack import generate_submission_materials


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", choices=["all", "metrics", "demo"], default="all")
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--submission-root", default=None)
    args = parser.parse_args()
    outputs = generate_submission_materials(
        repo_root(),
        section=args.section,
        reports_root=args.reports_root,
        submission_root=args.submission_root,
    )
    print(f"submission_section={args.section}")
    print(f"generated_file_count={len(outputs)}")
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
