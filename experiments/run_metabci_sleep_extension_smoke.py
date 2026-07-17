"""Smoke-test the installable ``metabci_sleep`` public API."""

from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
from metabci.brainda.datasets.base import BaseDataset
from metabci.brainda.paradigms.base import BaseParadigm

from metabci_sleep.algorithms import (
    CoverageRiskEvaluator,
    SleepFeatureExtractor,
    SleepMetrics,
    SleepQualityAuditor,
    TrustedRejector,
)
from metabci_sleep.brainstim import SleepCalibrationProtocol
from metabci_sleep.datasets import SleepEDF
from metabci_sleep.paradigms import SleepStaging
from metabci_sleep.realtime import OpenBCISleepWorker, WindowIntegrityAuditor
from metabci_sleep.reporting import SleepReportBuilder

from MetaSleepGuard.experiments.common import create_run_dir, write_json


class SyntheticSleepDataset(BaseDataset):
    """Small in-memory dataset used only for extension API verification."""

    def __init__(self) -> None:
        super().__init__(
            dataset_code="SyntheticSleep-MetaBCI",
            subjects=["S1", "S2"],
            events={
                "W": (1, (0.0, 30.0)),
                "N1": (2, (0.0, 30.0)),
                "N2": (3, (0.0, 30.0)),
                "N3": (4, (0.0, 30.0)),
                "REM": (5, (0.0, 30.0)),
            },
            channels=["CH1", "CH2"],
            srate=100,
            paradigm="sleep_staging",
        )

    def data_path(self, subject, **kwargs):
        del kwargs
        if subject not in self.subjects:
            raise ValueError(subject)
        return []

    def _get_single_subject_data(self, subject, verbose=None):
        del verbose
        if subject not in self.subjects:
            raise ValueError(subject)
        sfreq = 100.0
        epoch_samples = int(sfreq * 30)
        t = np.arange(epoch_samples * 5) / sfreq
        signals = np.vstack(
            [
                20e-6 * np.sin(2 * np.pi * 8 * t),
                18e-6 * np.sin(2 * np.pi * 10 * t),
            ]
        )
        raw = mne.io.RawArray(
            signals,
            mne.create_info(["CH1", "CH2"], sfreq=sfreq, ch_types=["eeg", "eeg"]),
            verbose="ERROR",
        )
        raw.set_annotations(
            mne.Annotations(
                onset=[0, 30, 60, 90, 120],
                duration=[30] * 5,
                description=["1", "2", "3", "4", "5"],
            )
        )
        return {"session_1": {"run_1": raw}}


def main() -> None:
    run_dir = create_run_dir("metabci_sleep_extension")
    dataset = SyntheticSleepDataset()
    paradigm = SleepStaging(task="3class", channels=["CH1", "CH2"])
    X, y, meta = paradigm.get_data(dataset, subjects=dataset.subjects, return_concat=True)

    auditor = SleepQualityAuditor(sfreq=dataset.srate, channel_names=dataset.channels)
    quality = auditor.transform(X[:2])
    rejector = TrustedRejector()
    accepted = rejector.decide([0.1, 0.8, 0.1], ["Wake", "NREM", "REM"], "A", True)
    rejected = rejector.decide([0.1, 0.8, 0.1], ["Wake", "NREM", "REM"], "D", False)
    features = SleepFeatureExtractor(dataset.srate, dataset.channels).transform(
        X, subject_ids=meta["subject"].tolist()
    )
    sleep_metrics = SleepMetrics().compute(meta["stage"].tolist())
    timestamps = np.arange(3000) / dataset.srate
    integrity = WindowIntegrityAuditor(dataset.srate).audit(timestamps, X[0])
    worker = OpenBCISleepWorker(sfreq=dataset.srate, channel_names=dataset.channels)
    worker.pre()
    worker_rows = worker.consume(X[0].T)
    report = SleepReportBuilder(run_dir / "report").build(worker_rows, {"source": "synthetic smoke"})
    portable_report = dict(report)
    for key in ("html", "markdown", "csv", "json"):
        portable_report[key] = str(Path(report[key]).relative_to(run_dir))
    marker_csv = SleepCalibrationProtocol().run(
        run_dir / "brainstim_markers.csv", dry_run=True, use_psychopy=False, countdown_sec=0
    )
    risk = CoverageRiskEvaluator().curve(
        [0, 1], np.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]), thresholds=[0.0, 0.9]
    )

    payload = {
        "distribution": "metabci-sleep",
        "import_package": "metabci_sleep",
        "official_metabci_modified": False,
        "components": {
            "SleepEDF": {
                "inherits": "metabci.brainda.datasets.base.BaseDataset",
                "issubclass": issubclass(SleepEDF, BaseDataset),
            },
            "SleepStaging": {
                "inherits": "metabci.brainda.paradigms.base.BaseParadigm",
                "issubclass": issubclass(SleepStaging, BaseParadigm),
            },
            "SleepQualityAuditor": {
                "wraps": "MetaSleepGuard.quality.quality_audit",
            },
            "TrustedRejector": {
                "wraps": "MetaSleepGuard.rejection.rejector.ActiveRejector",
            },
        },
        "synthetic_smoke": {
            "X_shape": list(X.shape),
            "y_shape": list(y.shape),
            "meta_columns": list(meta.columns),
            "subjects": sorted(meta["subject"].unique().tolist()),
            "stages": sorted(meta["stage"].unique().tolist()),
            "quality_rows": quality,
            "accepted_decision": accepted.__dict__,
            "rejected_decision": rejected.__dict__,
            "feature_shape": list(features.shape),
            "sleep_metrics": sleep_metrics,
            "integrity": integrity,
            "worker_rows": worker_rows,
            "report": portable_report,
            "brainstim_marker_csv": marker_csv.name,
            "coverage_risk": risk,
        },
    }
    output = write_json(payload, run_dir / "metabci_sleep_extension_smoke.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"extension_smoke_json={output}")
    if not payload["components"]["SleepEDF"]["issubclass"]:
        raise SystemExit("SleepEDF does not inherit MetaBCI BaseDataset")
    if not payload["components"]["SleepStaging"]["issubclass"]:
        raise SystemExit("SleepStaging does not inherit MetaBCI BaseParadigm")
    if X.shape != (10, 2, 3000):
        raise SystemExit(f"unexpected extension data shape: {X.shape}")
    if not accepted.accepted or rejected.accepted:
        raise SystemExit("trusted rejection smoke failed")
    if not report["html"] or not marker_csv.exists():
        raise SystemExit("report or Brainstim extension smoke failed")


if __name__ == "__main__":
    main()
