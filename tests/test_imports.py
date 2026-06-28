def test_core_imports():
    import MetaSleepGuard
    from MetaSleepGuard.datasets.public_sleep import generate_synthetic_public_records
    from MetaSleepGuard.models.train_xgb import build_classifier
    from MetaSleepGuard.quality import audit_epoch
    from MetaSleepGuard.rejection import ActiveRejector

    assert MetaSleepGuard.__version__
    assert generate_synthetic_public_records
    assert build_classifier
    assert audit_epoch
    assert ActiveRejector


def test_three_metabci_component_statuses():
    from MetaSleepGuard.metabci_integration import inspect_metabci_components

    statuses = inspect_metabci_components()
    assert set(statuses) == {"brainda", "brainflow", "brainstim"}
