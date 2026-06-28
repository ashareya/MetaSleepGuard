"""BrainFlow interface for OpenBCI Cyton plus a no-hardware synthetic stream."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class StreamChunk:
    data: np.ndarray
    sfreq: float
    channel_names: list[str]
    expected_samples: int | None = None
    received_samples: int | None = None


class OpenBCICytonStream:
    """OpenBCI Cyton reader using BrainFlow."""

    def __init__(self, serial_port: str, channel_indices: Sequence[int] = (1, 6), board_id: int | None = None) -> None:
        try:
            from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
        except Exception as exc:  # pragma: no cover - optional hardware dependency
            raise RuntimeError("brainflow is required for OpenBCI real-time acquisition") from exc
        self.BoardShim = BoardShim
        self.board_id = int(board_id if board_id is not None else BoardIds.CYTON_BOARD.value)
        params = BrainFlowInputParams()
        params.serial_port = serial_port
        self.board = BoardShim(self.board_id, params)
        self.channel_indices = list(channel_indices)
        self.sfreq = float(BoardShim.get_sampling_rate(self.board_id))
        eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.eeg_channels = [eeg_channels[i] for i in self.channel_indices]
        self.channel_names = [f"Ch{i + 1}" for i in self.channel_indices]
        self.prepared = False
        self.streaming = False

    def start(self) -> None:
        LOGGER.info("Preparing OpenBCI Cyton session")
        self.board.prepare_session()
        self.prepared = True
        self.board.start_stream()
        self.streaming = True

    def read(self, seconds: float = 1.0) -> StreamChunk:
        n_samples = int(round(self.sfreq * seconds))
        data = self.board.get_board_data(n_samples)
        eeg_volts = data[self.eeg_channels, -n_samples:] * 1e-6
        received = eeg_volts.shape[1]
        if received < n_samples:
            LOGGER.warning("OpenBCI chunk missing %d of %d expected samples", n_samples - received, n_samples)
            padded = np.full((len(self.eeg_channels), n_samples), np.nan, dtype=float)
            if received:
                padded[:, -received:] = eeg_volts
            eeg_volts = padded
        return StreamChunk(
            data=eeg_volts,
            sfreq=self.sfreq,
            channel_names=self.channel_names,
            expected_samples=n_samples,
            received_samples=received,
        )

    def stop(self) -> None:
        try:
            if self.streaming:
                self.board.stop_stream()
        finally:
            self.streaming = False
            if self.prepared:
                self.board.release_session()
                self.prepared = False


class SyntheticBrainFlowStream:
    """No-hardware 250 Hz stream for tests and demos."""

    def __init__(self, sfreq: float = 250.0, channel_names: Sequence[str] = ("Ch2", "Ch7"), seed: int = 11) -> None:
        self.sfreq = float(sfreq)
        self.channel_names = list(channel_names)
        self.rng = np.random.default_rng(seed)
        self.sample_index = 0

    def start(self) -> None:
        self.sample_index = 0

    def read(self, seconds: float = 1.0) -> StreamChunk:
        n_samples = int(round(self.sfreq * seconds))
        t = (np.arange(n_samples) + self.sample_index) / self.sfreq
        ch1 = 20e-6 * np.sin(2 * np.pi * 10.0 * t) + self.rng.normal(scale=4e-6, size=n_samples)
        ch2 = 18e-6 * np.sin(2 * np.pi * 6.0 * t + 0.2) + self.rng.normal(scale=4e-6, size=n_samples)
        self.sample_index += n_samples
        return StreamChunk(np.vstack([ch1, ch2]), self.sfreq, self.channel_names, n_samples, n_samples)

    def stop(self) -> None:
        return None
