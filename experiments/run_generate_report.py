"""Generate a demo report from synthetic runtime rows."""

from __future__ import annotations

from MetaSleepGuard.experiments.common import create_run_dir
from MetaSleepGuard.reports.report_generator import generate_html_report


def main() -> None:
    rows = [
        {
            "window_start_time": 0,
            "window_end_time": 30,
            "stage": "W",
            "confidence": 0.82,
            "accepted": True,
            "quality_grade": "A",
            "bad_flags": "",
            "reason": "accepted",
        },
        {
            "window_start_time": 30,
            "window_end_time": 60,
            "stage": "暂不判定",
            "confidence": 0.38,
            "accepted": False,
            "quality_grade": "D",
            "bad_flags": "motion_artifact",
            "reason": "low_signal_quality",
        },
    ]
    path = generate_html_report(
        create_run_dir("reports") / "demo_report.html",
        {"mode": "demo", "device": "synthetic", "note": "占位演示报告，不代表真实准确率"},
        rows,
    )
    print(f"report={path}")


if __name__ == "__main__":
    main()
