import numpy as np
from pathlib import Path

from MetaSleepGuard.datasets.public_sleep.loaders import _sleep_edf_subject_id, _subject_from_path, find_sleep_edf_pairs
from MetaSleepGuard.preprocessing.epoching import align_epoch_labels, epoch_signal, subject_level_split


def test_epoch_signal_30_seconds():
    sfreq = 250
    signals = np.zeros((2, sfreq * 65))
    epochs = epoch_signal(signals, sfreq)
    assert epochs.shape == (2, 2, sfreq * 30)


def test_align_labels_and_subject_split():
    labels = align_epoch_labels(["W"], 3)
    assert labels == ["W", "UNKNOWN", "UNKNOWN"]
    train, test = subject_level_split(["s1", "s1", "s2", "s2"], test_fraction=0.5, seed=1)
    assert train.shape == test.shape == (4,)
    assert not any(train & test)


def test_sleep_edf_nights_share_subject_id():
    assert _sleep_edf_subject_id(Path("SC4001E0-PSG.edf")) == "SC400"
    assert _sleep_edf_subject_id(Path("SC4002E0-PSG.edf")) == "SC400"


def test_sleep_edf_standard_psg_hypnogram_pairing(tmp_path):
    psg = tmp_path / "SC4001E0-PSG.edf"
    hypnogram = tmp_path / "SC4001EC-Hypnogram.edf"
    psg.touch()
    hypnogram.touch()
    assert find_sleep_edf_pairs(tmp_path) == [(psg, hypnogram)]


def test_isruc_recordings_use_subject_directory():
    first = Path("ISRUC/SubgroupI/subject01/night_1.edf")
    second = Path("ISRUC/SubgroupI/subject01/night_2.edf")
    assert _subject_from_path(first) == _subject_from_path(second) == "subject01"
