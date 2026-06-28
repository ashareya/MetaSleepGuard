"""Replay an OpenBCI CSV/TXT or BDF/FIF file through the runtime pipeline."""

from __future__ import annotations

import argparse

from MetaSleepGuard.experiments.common import create_run_dir, setup_logging
from MetaSleepGuard.realtime.openbci_file_loader import load_replay_file
from MetaSleepGuard.realtime.realtime_pipeline import replay_array
from MetaSleepGuard.reports.report_generator import generate_html_report
from MetaSleepGuard.visualization.plot_hypnogram import plot_hypnogram
from MetaSleepGuard.visualization.plot_quality import plot_confidence_trend, plot_quality_trend
from MetaSleepGuard.visualization.plot_spectrum import plot_alpha_comparison, plot_spectrum
from MetaSleepGuard.visualization.plot_waveform import plot_waveform
from MetaSleepGuard.visualization.dashboard import write_static_dashboard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--sfreq", type=float, default=250.0)
    parser.add_argument("--units", default="microvolts", choices=["microvolts", "volts"])
    parser.add_argument("--channels", nargs=2, default=None)
    args = parser.parse_args()
    setup_logging()
    data = load_replay_file(args.file, sfreq=args.sfreq, units=args.units, channels=args.channels)
    run_dir = create_run_dir("file_replay")
    rows = replay_array(data.signals, data.sfreq, data.channel_names, model_bundle=args.model, output_log=run_dir / "runtime_log.csv")
    figures = [
        plot_waveform(data.signals, data.sfreq, data.channel_names, run_dir / "waveform.png"),
        plot_spectrum(data.signals, data.sfreq, data.channel_names, run_dir / "spectrum.png"),
        plot_alpha_comparison(data.signals, data.sfreq, data.channel_names, run_dir / "alpha.png"),
        plot_hypnogram([row["stage"] for row in rows], run_dir / "hypnogram.png"),
        plot_quality_trend([row["quality_grade"] for row in rows], run_dir / "quality.png"),
        plot_confidence_trend([row["confidence"] for row in rows], [row["accepted"] for row in rows], run_dir / "confidence.png"),
    ]
    report = generate_html_report(
        run_dir / "file_replay_report.html",
        {
            "mode": "file_replay",
            "source": args.file,
            "device_or_format": data.metadata.get("source", "unknown"),
            "sfreq_hz": data.sfreq,
            "channels": ", ".join(data.channel_names),
            "recording_duration_sec": data.signals.shape[1] / data.sfreq,
            "model": args.model or "not loaded",
        },
        rows,
        figures=[str(path) for path in figures],
    )
    dashboard = write_static_dashboard(rows, [str(path) for path in figures], run_dir / "dashboard.html")
    print(f"dashboard={dashboard}")
    print(f"report={report}")


if __name__ == "__main__":
    main()
