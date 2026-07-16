"""Synthetic end-to-end demonstration of the biomedical extension APIs."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from metabci_sleep.algorithms import (
    SleepMetrics,
    SleepQualityAuditor,
    SleepStagingEstimator,
    TrustedRejector,
)
from metabci_sleep.reporting import SleepReportBuilder


def main() -> None:
    rng = np.random.default_rng(7)
    epochs = rng.normal(size=(30, 2, 3000)) * 20  # microvolts
    labels = np.asarray([0, 1, 2] * 10)
    subjects = ["S1"] * 15 + ["S2"] * 15
    estimator = SleepStagingEstimator(
        task="3class",
        sfreq=100,
        channel_names=["Fp1", "Fp2"],
        estimator=RandomForestClassifier(n_estimators=20, random_state=7),
    ).fit(epochs, labels, subject_ids=subjects)
    probabilities = estimator.predict_proba(epochs, subject_ids=subjects)
    quality = SleepQualityAuditor(100, ["Fp1", "Fp2"])
    rejector = TrustedRejector()
    rows = []
    for index, epoch in enumerate(epochs):
        audited = quality.audit_epoch(epoch * 1e-6, start_time=index * 30)
        decision = rejector.decide(
            probabilities[index],
            estimator.classes,
            audited.quality_grade,
            audited.is_reliable,
        )
        rows.append(
            {
                "window_start_time": index * 30,
                "window_end_time": (index + 1) * 30,
                "stage": decision.stage,
                "confidence": decision.confidence,
                "accepted": decision.accepted,
                "quality_grade": audited.quality_grade,
                "bad_flags": "|".join(audited.bad_flags),
                "reason": decision.reason,
            }
        )
    print(SleepMetrics().compute([row["stage"] for row in rows]))
    print(SleepReportBuilder("biomedical_extension_demo").build(rows, {"source": "synthetic"}))


if __name__ == "__main__":
    main()
