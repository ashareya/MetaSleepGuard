"""MetaBCI ``ProcessWorker`` adapter for the existing realtime pipeline."""

from __future__ import annotations

from typing import Callable

import numpy as np
from metabci.brainflow.workers import ProcessWorker

from MetaSleepGuard.realtime.realtime_pipeline import RealtimePipeline


class OpenBCISleepWorker(ProcessWorker):
    def __init__(
        self,
        sfreq: float = 250.0,
        channel_names=("Ch2", "Ch7"),
        model_bundle=None,
        input_layout: str = "samples_by_channels",
        callback: Callable[[dict], None] | None = None,
        timeout: float = 1e-3,
    ) -> None:
        super().__init__(timeout=timeout, name="metabci_sleep")
        if input_layout not in {"samples_by_channels", "channels_by_samples"}:
            raise ValueError("unsupported input_layout")
        self.pipeline = RealtimePipeline(
            sfreq=sfreq,
            channel_names=channel_names,
            model_bundle=model_bundle,
        )
        self.input_layout = input_layout
        self.callback = callback
        self.results: list[dict] = []

    def pre(self):
        self.results = []

    def consume(self, data):
        array = np.asarray(data, dtype=float)
        if array.ndim != 2:
            raise ValueError("worker data must be two-dimensional")
        if self.input_layout == "samples_by_channels":
            n_channels = len(self.pipeline.channel_names)
            if array.shape[1] == n_channels + 1:
                array = array[:, :n_channels]
            array = array.T
        outputs = self.pipeline.append_and_process(array)
        self.results.extend(outputs)
        if self.callback:
            for row in outputs:
                self.callback(row)
        return outputs

    def post(self):
        return None
