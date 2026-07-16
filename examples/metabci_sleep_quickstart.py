"""Minimal real-API example for the MetaBCI-compatible sleep extension."""

from __future__ import annotations

from pathlib import Path

from metabci_sleep.algorithms import SleepQualityAuditor, TrustedRejector
from metabci_sleep.datasets import SleepEDF
from metabci_sleep.paradigms import SleepStaging


def main() -> None:
    root = Path("data/public_sleep/sleep_edf_raw/physionet-sleep-data")
    dataset = SleepEDF(root, channels=["EEG Fpz-Cz", "EEG Pz-Oz"], max_subjects=1)
    paradigm = SleepStaging(task="3class")
    X, y, meta = paradigm.get_data(dataset, subjects=dataset.subjects)

    auditor = SleepQualityAuditor(dataset.srate, dataset.channels)
    quality = auditor.audit_epoch(X[0] * 1e-6)
    rejector = TrustedRejector()
    decision = rejector.decide(
        probabilities=[0.1, 0.8, 0.1],
        classes=["Wake", "NREM", "REM"],
        quality_grade=quality.quality_grade,
        is_reliable=quality.is_reliable,
    )
    print("epochs:", X.shape)
    print("labels:", y.shape)
    print("metadata columns:", list(meta.columns))
    print("quality:", quality.to_row())
    print("decision:", decision)


if __name__ == "__main__":
    main()
