from pathlib import Path

import mne
import numpy as np


def _synthetic_dataset():
    from metabci.brainda.datasets.base import BaseDataset

    class Dataset(BaseDataset):
        def __init__(self):
            super().__init__(
                dataset_code="SyntheticSleep-Test",
                subjects=["S1"],
                events={
                    "W": (1, (0.0, 30.0)),
                    "N1": (2, (0.0, 30.0)),
                    "N2": (3, (0.0, 30.0)),
                    "N3": (4, (0.0, 30.0)),
                    "REM": (5, (0.0, 30.0)),
                },
                channels=["CH1", "CH2"],
                srate=10,
                paradigm="sleep_staging",
            )

        def data_path(self, subject, **kwargs):
            return []

        def _get_single_subject_data(self, subject, verbose=None):
            del subject, verbose
            data = np.zeros((2, 5 * 300))
            raw = mne.io.RawArray(
                data,
                mne.create_info(["CH1", "CH2"], 10, ["eeg", "eeg"]),
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

    return Dataset()


def test_metabci_sleep_public_api_and_inheritance():
    from metabci.brainda.datasets.base import BaseDataset
    from metabci.brainda.paradigms.base import BaseParadigm
    from metabci_sleep import SleepEDF, SleepQualityAuditor, SleepStaging, TrustedRejector

    assert issubclass(SleepEDF, BaseDataset)
    assert issubclass(SleepStaging, BaseParadigm)
    assert SleepQualityAuditor
    assert TrustedRejector


def test_sleep_staging_three_and_five_class_outputs():
    from metabci_sleep.paradigms import SleepStaging

    dataset = _synthetic_dataset()
    X5, y5, meta5 = SleepStaging("5class").get_data(dataset)
    X3, y3, meta3 = SleepStaging("3class").get_data(dataset)
    assert X5.shape == (5, 2, 300)
    assert y5.tolist() == [0, 1, 2, 3, 4]
    assert meta5["stage"].tolist() == ["W", "N1", "N2", "N3", "REM"]
    assert X3.shape == (5, 2, 300)
    assert y3.tolist() == [0, 1, 1, 1, 2]
    assert meta3["stage"].tolist() == ["W", "NREM", "NREM", "NREM", "REM"]
    assert {"subject", "session", "run", "stage", "epoch_index", "dataset"} <= set(meta3)


def test_sleep_staging_rejects_incompatible_dataset():
    from metabci_sleep.paradigms import SleepStaging

    dataset = _synthetic_dataset()
    dataset.paradigm = "imagery"
    try:
        SleepStaging().get_data(dataset)
    except TypeError as exc:
        assert "sleep_staging" in str(exc)
    else:
        raise AssertionError("incompatible dataset should be rejected")


def test_quality_and_rejection_wrappers_match_core():
    from MetaSleepGuard.quality import audit_epoch
    from metabci_sleep.algorithms import SleepQualityAuditor, TrustedRejector

    epoch = np.zeros((2, 300))
    core = audit_epoch(epoch, sfreq=10, channel_names=["CH1", "CH2"])
    wrapped = SleepQualityAuditor(10, ["CH1", "CH2"]).audit_epoch(epoch)
    assert wrapped.to_row() == core.to_row()

    rejector = TrustedRejector()
    accepted = rejector.decide([0.1, 0.8, 0.1], ["Wake", "NREM", "REM"], "A", True)
    rejected = rejector.decide([0.1, 0.8, 0.1], ["Wake", "NREM", "REM"], "D", False)
    assert accepted.accepted and accepted.stage == "NREM"
    assert not rejected.accepted and rejected.stage == "暂不判定"


def test_sleep_edf_dataset_discovers_real_pairs_and_standard_paths():
    from metabci.brainda.datasets.base import BaseDataset
    from metabci_sleep.datasets import SleepEDF

    root = Path(__file__).resolve().parents[1] / "data" / "public_sleep" / "sleep_edf_raw" / "physionet-sleep-data"
    dataset = SleepEDF(root, max_subjects=1)
    assert isinstance(dataset, BaseDataset)
    assert dataset.paradigm == "sleep_staging"
    assert dataset.subjects == ["SC400"]
    paths = dataset.data_path("SC400")
    assert len(paths) == 1
    assert len(paths[0]) == 2
    assert all(Path(path).exists() for path in paths[0])


def test_sleep_edf_returns_standard_nested_raw_and_expands_annotations(tmp_path):
    from unittest.mock import patch

    from metabci_sleep.datasets import SleepEDF

    psg = tmp_path / "SC4001E0-PSG.edf"
    hypnogram = tmp_path / "SC4001EC-Hypnogram.edf"
    psg.touch()
    hypnogram.touch()
    raw = mne.io.RawArray(
        np.zeros((2, 1200)),
        mne.create_info(["EEG Fpz-Cz", "EEG Pz-Oz"], 10, ["eeg", "eeg"]),
        verbose="ERROR",
    )
    annotations = mne.Annotations(
        onset=[0, 60, 90],
        duration=[60, 30, 30],
        description=["Sleep stage W", "Sleep stage 2", "Movement time"],
    )
    with (
        patch(
            "metabci_sleep.datasets.sleep_edf.find_sleep_edf_pairs",
            return_value=[(psg, hypnogram)],
        ),
        patch(
            "metabci_sleep.datasets.sleep_edf.mne.io.read_raw_edf",
            side_effect=lambda *args, **kwargs: raw.copy(),
        ),
        patch(
            "metabci_sleep.datasets.sleep_edf.mne.read_annotations",
            return_value=annotations,
        ),
    ):
        dataset = SleepEDF(tmp_path)
        nested = dataset.get_data(dataset.subjects)
    returned = nested["SC400"]["session_1"]["run_1"]
    assert isinstance(returned, mne.io.BaseRaw)
    assert returned.ch_names == ["EEG Fpz-Cz", "EEG Pz-Oz"]
    assert returned.annotations.onset.tolist() == [0.0, 30.0, 60.0]
    assert returned.annotations.description.tolist() == ["1", "1", "3"]
