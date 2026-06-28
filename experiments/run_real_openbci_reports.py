"""Generate formal reports from the SX real OpenBCI experiment files."""

from __future__ import annotations

import argparse

from MetaSleepGuard.experiments.common import repo_root
from MetaSleepGuard.reports.real_openbci_reports import generate_real_openbci_reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs-root",
        default=str(repo_root().parent / "metasleepguard_protocol_timer" / "logs"),
        help="Directory containing the SX 10-minute and 60-minute protocol log folders",
    )
    parser.add_argument(
        "--output-root",
        default=str(repo_root() / "_codex_tmp" / "metasleepguard_outputs" / "real_openbci_reports"),
    )
    args = parser.parse_args()
    results = generate_real_openbci_reports(args.logs_root, args.output_root)
    ten = results["ten_minute"]
    sixty = results["sixty_minute"]
    print(f"ten_min_formal_file={ten['summary']['formal_data_file']}")
    print(f"ten_min_debug_file={ten['summary']['debug_short_file']}")
    print(f"ten_min_coverage_ratio={ten['summary']['coverage_ratio']:.6f}")
    print(f"sixty_min_segment_count={sixty['summary']['segment_count']}")
    print(f"sixty_min_coverage_ratio={sixty['summary']['coverage_ratio']:.6f}")
    print(f"sixty_min_missing_duration_sec={sixty['summary']['missing_duration_sec']:.3f}")
    print(f"affected_windows={sixty['summary']['affected_window_indices']}")
    print(f"ten_min_report={ten['report_html']}")
    print(f"sixty_min_report={sixty['report_html']}")
    print(f"real_data_readme={results['readme']}")


if __name__ == "__main__":
    main()
