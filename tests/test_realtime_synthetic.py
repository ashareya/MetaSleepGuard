import numpy as np

from MetaSleepGuard.realtime.openbci_brainflow_stream import SyntheticBrainFlowStream
from MetaSleepGuard.realtime.realtime_pipeline import RealtimePipeline
from MetaSleepGuard.realtime.ring_buffer import RingBuffer


def test_ring_buffer_and_synthetic_pipeline():
    stream = SyntheticBrainFlowStream()
    stream.start()
    chunk = stream.read(seconds=31)
    assert chunk.data.shape == (2, 250 * 31)
    buf = RingBuffer(2, 250 * 40)
    buf.append(chunk.data)
    assert buf.ready(250 * 30)
    pipeline = RealtimePipeline(sfreq=chunk.sfreq, channel_names=chunk.channel_names)
    rows = pipeline.append_and_process(chunk.data)
    assert len(rows) == 1
    assert rows[0]["stage"] == "暂不判定"


def test_large_chunk_processes_first_epoch_not_trailing_epoch():
    sfreq = 10
    pipeline = RealtimePipeline(sfreq=sfreq, epoch_sec=2, channel_names=["A", "B"])
    samples = np.vstack([np.arange(30), np.arange(30) + 100]).astype(float)
    captured = []

    def capture(epoch, start_time=0.0):
        captured.append((epoch.copy(), start_time))
        return {"window_start_time": start_time}

    pipeline.process_epoch = capture
    pipeline.append_and_process(samples)
    assert np.array_equal(captured[0][0], samples[:, :20])
    assert captured[0][1] == 0.0
