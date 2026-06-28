import numpy as np

from MetaSleepGuard.features import extract_features_from_epochs
from MetaSleepGuard.features.causal_context import append_causal_context
from MetaSleepGuard.datasets.public_sleep import generate_synthetic_public_records
from MetaSleepGuard.models.train_xgb import records_to_feature_dataset
from MetaSleepGuard.preprocessing.epoching import epoch_signal


def test_feature_dimensions_and_causal_context():
    sfreq = 250
    t = np.arange(sfreq * 30) / sfreq
    epoch = np.vstack([np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 6 * t)])
    epochs = np.stack([epoch, epoch * 0.5, epoch * 1.5])
    X, names = extract_features_from_epochs(epochs, sfreq, ["Ch2", "Ch7"])
    assert X.shape[0] == 3
    assert X.shape[1] == len(names)
    assert "ch1_ch2_corr" in names
    Xc = append_causal_context(X, ["s1", "s1", "s1"], history=2)
    assert Xc.shape == (3, X.shape[1] * 3)
    assert np.allclose(Xc[0, : X.shape[1] * 2], 0.0)


def test_context_uses_immediately_previous_unknown_epoch():
    record = generate_synthetic_public_records(n_subjects=1, n_epochs=4)[0]
    record.labels = ["W", "UNKNOWN", "N1", "N2"]
    data = records_to_feature_dataset([record], apply_preprocessing=False)
    epochs = epoch_signal(record.signals, record.sfreq)
    base, _ = extract_features_from_epochs(epochs, record.sfreq, ["EEG1", "EEG2"])
    width = base.shape[1]
    assert np.allclose(data.X[1, width : 2 * width], base[1])
