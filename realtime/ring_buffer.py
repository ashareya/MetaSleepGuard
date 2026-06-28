"""Fixed-size multi-channel ring buffer."""

from __future__ import annotations

import numpy as np


class RingBuffer:
    def __init__(self, n_channels: int, max_samples: int) -> None:
        self.n_channels = int(n_channels)
        self.max_samples = int(max_samples)
        self.buffer = np.zeros((self.n_channels, self.max_samples), dtype=float)
        self.write_index = 0
        self.samples_seen = 0

    def append(self, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=float)
        if samples.ndim == 1:
            samples = samples.reshape(self.n_channels, 1)
        if samples.shape[0] != self.n_channels:
            raise ValueError("sample channel count does not match ring buffer")
        for col in range(samples.shape[1]):
            self.buffer[:, self.write_index] = samples[:, col]
            self.write_index = (self.write_index + 1) % self.max_samples
            self.samples_seen += 1

    def ready(self, n_samples: int) -> bool:
        return self.samples_seen >= n_samples

    def get_last(self, n_samples: int) -> np.ndarray:
        if not self.ready(n_samples):
            raise ValueError("not enough samples in ring buffer")
        indices = (np.arange(n_samples) + self.write_index - n_samples) % self.max_samples
        return self.buffer[:, indices].copy()

    def get_all_ordered(self) -> np.ndarray:
        count = min(self.samples_seen, self.max_samples)
        indices = (np.arange(count) + self.write_index - count) % self.max_samples
        return self.buffer[:, indices].copy()

