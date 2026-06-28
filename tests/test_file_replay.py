import numpy as np
import pandas as pd

from MetaSleepGuard.realtime.openbci_file_loader import load_mne_file, load_openbci_csv


def test_openbci_csv_loader(tmp_path):
    sfreq = 250
    t = np.arange(sfreq * 2) / sfreq
    path = tmp_path / "openbci.csv"
    pd.DataFrame(
        {
            "Sample Index": range(t.size),
            "EXG Channel 0": np.sin(2 * np.pi * 10 * t),
            "EXG Channel 1": np.sin(2 * np.pi * 6 * t),
            "EXG Channel 6": np.sin(2 * np.pi * 3 * t),
        }
    ).to_csv(path, index=False)
    data = load_openbci_csv(path, channel_indices=(1, 2), sfreq=sfreq)
    assert data.signals.shape == (2, t.size)
    assert len(data.channel_names) == 2
    assert np.max(np.abs(data.signals)) <= 1.01e-6


def test_mne_replay_selects_exactly_two_channels(tmp_path):
    import mne

    sfreq = 100
    signals = np.zeros((3, sfreq * 2))
    raw = mne.io.RawArray(
        signals,
        mne.create_info(["Fp1", "Fp2", "C3"], sfreq, ch_types=["eeg"] * 3),
        verbose="ERROR",
    )
    path = tmp_path / "three_channels_raw.fif"
    raw.save(path, overwrite=True, verbose="ERROR")
    replay = load_mne_file(path)
    assert replay.signals.shape == (2, sfreq * 2)
    assert replay.channel_names == ["Fp1", "Fp2"]
