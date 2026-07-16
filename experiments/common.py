"""Shared experiment utilities."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import logging

import yaml

from MetaSleepGuard.datasets.public_sleep import (
    generate_synthetic_public_records,
    load_isruc_sleep,
    load_sleep_edf,
)

LOGGER = logging.getLogger(__name__)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    """Return the Git repository root containing ``pyproject.toml``."""

    return project_root()


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else project_root() / "config.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def output_dir(config: dict | None = None) -> Path:
    config = config or load_config()
    configured = config.get("paths", {}).get("output_dir") if isinstance(config.get("paths"), dict) else None
    return (repo_root() / configured).resolve() if configured else (repo_root() / "outputs" / "metasleepguard_outputs").resolve()


def create_run_dir(category: str) -> Path:
    """Create a unique output directory so one recording never overwrites another."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = output_dir() / category / timestamp
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_public_records(
    dataset: str,
    root: str | Path | None = None,
    max_subjects: int | None = None,
    allow_synthetic: bool = True,
):
    dataset_key = dataset.lower()
    if root is None:
        config = load_config()
        path_key = "sleep_edf_root" if dataset_key in {"sleep-edf", "sleep_edf", "sleepedf"} else "isruc_root"
        configured = config.get("paths", {}).get(path_key, "")
        if configured:
            root = project_root() / configured
    if root and Path(root).exists():
        if dataset_key in {"sleep-edf", "sleep_edf", "sleepedf"}:
            records = load_sleep_edf(root, max_subjects=max_subjects)
        elif dataset_key in {"isruc", "isruc-sleep", "isruc_sleep"}:
            records = load_isruc_sleep(root, max_subjects=max_subjects)
        else:
            raise ValueError(f"unknown dataset: {dataset}")
        if records:
            return records
        if not allow_synthetic:
            raise RuntimeError(
                f"No usable {dataset} recordings and expert labels were found in {Path(root).resolve()}. "
                "Place the public dataset there or pass the correct dataset root. Synthetic fallback is disabled."
            )
        LOGGER.warning("No records found in %s; falling back to synthetic demo records", root)
    else:
        if not allow_synthetic:
            raise RuntimeError(
                f"No usable {dataset} root was configured. Pass the public dataset root; "
                "real OpenBCI data cannot be used for sleep-staging accuracy."
            )
        LOGGER.warning("No usable %s root was configured; using synthetic demo records", dataset)
    return generate_synthetic_public_records(dataset=dataset_key, n_subjects=max_subjects or 4)


def write_json(data: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(levelname)s %(name)s: %(message)s")
