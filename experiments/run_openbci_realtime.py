"""Run OpenBCI Cyton real-time acquisition or synthetic no-hardware mode."""

from __future__ import annotations

import argparse
import time

from MetaSleepGuard.experiments.common import create_run_dir, setup_logging
from MetaSleepGuard.realtime.openbci_brainflow_stream import OpenBCICytonStream, SyntheticBrainFlowStream
from MetaSleepGuard.realtime.realtime_pipeline import RealtimePipeline
from MetaSleepGuard.realtime.save_raw import RawCsvWriter
from MetaSleepGuard.reports.report_generator import generate_html_report
from MetaSleepGuard.visualization.plot_hypnogram import plot_hypnogram
from MetaSleepGuard.visualization.plot_quality import plot_confidence_trend, plot_quality_trend
from MetaSleepGuard.visualization.plot_spectrum import plot_spectrum
from MetaSleepGuard.visualization.plot_waveform import plot_waveform
from MetaSleepGuard.visualization.dashboard import write_static_dashboard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial-port", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    if args.duration_sec <= 0:
        raise SystemExit("--duration-sec must be positive")
    setup_logging()
    if args.synthetic:
        stream = SyntheticBrainFlowStream()
    else:
        if not args.serial_port:
            raise SystemExit("--serial-port is required unless --synthetic is set")
        stream = OpenBCICytonStream(args.serial_port)
    run_dir = create_run_dir("openbci_realtime")
    pipeline = RealtimePipeline(
        sfreq=stream.sfreq,
        channel_names=stream.channel_names,
        model_bundle=args.model,
        output_log=run_dir / "runtime_log.csv",
    )
    raw_path = run_dir / "raw_eeg.csv"
    raw_writer = RawCsvWriter(raw_path, stream.sfreq, stream.channel_names)
    rows: list[dict] = []
    interrupted = False
    try:
        stream.start()
        acquired_sec = 0.0
        while acquired_sec < args.duration_sec:
            read_sec = min(1.0, args.duration_sec - acquired_sec)
            if not args.synthetic:
                time.sleep(read_sec)
            chunk = stream.read(seconds=read_sec)
            raw_writer.append(chunk.data)
            new_rows = pipeline.append_and_process(chunk.data)
            rows.extend(new_rows)
            for row in new_rows:
                print(row)
            if new_rows:
                _write_live_dashboard(run_dir, pipeline, stream.sfreq, stream.channel_names, rows)
            acquired_sec += read_sec
    except KeyboardInterrupt:
        interrupted = True
        print("recording_interrupted=True")
    finally:
        stream.stop()
    figures = []
    buffered = pipeline.buffer.get_all_ordered()
    if buffered.shape[1] > 0:
        figures.extend(
            [
                plot_waveform(buffered, stream.sfreq, stream.channel_names, run_dir / "waveform.png"),
                plot_spectrum(buffered, stream.sfreq, stream.channel_names, run_dir / "spectrum.png"),
            ]
        )
    if rows:
        figures.extend(
            [
                plot_hypnogram([row["stage"] for row in rows], run_dir / "hypnogram.png"),
                plot_quality_trend([row["quality_grade"] for row in rows], run_dir / "quality.png"),
                plot_confidence_trend(
                    [row["confidence"] for row in rows],
                    [row["accepted"] for row in rows],
                    run_dir / "confidence.png",
                ),
            ]
        )
    report = generate_html_report(
        run_dir / "realtime_report.html",
        {
            "mode": "synthetic_realtime" if args.synthetic else "openbci_cyton_realtime",
            "device": "SyntheticBrainFlowStream" if args.synthetic else "OpenBCI Cyton",
            "channels": ", ".join(stream.channel_names),
            "sfreq_hz": stream.sfreq,
            "recording_duration_sec": raw_writer.samples_written / stream.sfreq,
            "model": args.model or "not loaded",
            "model_synthetic_demo": bool(
                pipeline.inference and pipeline.inference.bundle.get("metadata", {}).get("synthetic_demo")
            ),
            "recording_interrupted": interrupted,
        },
        rows,
        figures=[str(path) for path in figures],
    )
    dashboard = write_static_dashboard(rows, [str(path) for path in figures], run_dir / "dashboard.html")
    print(f"raw_csv={raw_path}")
    print(f"runtime_log={run_dir / 'runtime_log.csv'}")
    print(f"dashboard={dashboard}")
    print(f"report={report}")


def _write_live_dashboard(run_dir, pipeline, sfreq, channel_names, rows) -> None:
    buffered = pipeline.buffer.get_all_ordered()
    figures = [
        plot_waveform(buffered, sfreq, channel_names, run_dir / "waveform.png"),
        plot_spectrum(buffered, sfreq, channel_names, run_dir / "spectrum.png"),
        plot_hypnogram([row["stage"] for row in rows], run_dir / "hypnogram.png"),
        plot_quality_trend([row["quality_grade"] for row in rows], run_dir / "quality.png"),
        plot_confidence_trend(
            [row["confidence"] for row in rows],
            [row["accepted"] for row in rows],
            run_dir / "confidence.png",
        ),
    ]
    write_static_dashboard(rows, [str(path) for path in figures], run_dir / "dashboard.html")


if __name__ == "__main__":
    main()
