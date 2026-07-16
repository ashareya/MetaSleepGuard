"""OpenBCI-style samples through the MetaBCI ProcessWorker-compatible API."""

from __future__ import annotations

import numpy as np

from metabci_sleep.realtime import OpenBCISleepWorker, WindowIntegrityAuditor


def main() -> None:
    sfreq = 250
    timestamps = np.arange(sfreq * 30) / sfreq
    samples = np.column_stack(
        [
            20e-6 * np.sin(2 * np.pi * 8 * timestamps),
            18e-6 * np.sin(2 * np.pi * 10 * timestamps),
        ]
    )
    worker = OpenBCISleepWorker(sfreq=sfreq, channel_names=["Ch2", "Ch7"])
    worker.pre()
    print(worker.consume(samples))
    print(WindowIntegrityAuditor(sfreq).audit(timestamps, samples.T * 1e6))


if __name__ == "__main__":
    main()
