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


def test_package_declares_runtime_dependencies():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8").lower()
    for dependency in ("scipy", "scikit-learn", "joblib", "matplotlib", "pyyaml"):
        assert f'"{dependency}>=' in text


def test_metabci_status_check_does_not_leave_log_file(tmp_path):
    import subprocess
    import sys

    code = (
        "from MetaSleepGuard.metabci_integration import inspect_metabci_components; "
        "inspect_metabci_components()"
    )
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)
    assert not (tmp_path / "log.txt").exists()
