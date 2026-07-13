"""Adapter between public sleep evaluation and MetaBCI Brainda primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
from typing import Sequence

import numpy as np

from .metabci_component_check import inspect_metabci_components


@dataclass(frozen=True)
class BraindaAdapterResult:
    ok: bool
    metabci_brainda_available: bool
    split_method: str
    n_splits: int
    held_out_subjects: list[str]
    no_subject_overlap: bool
    entrypoints: list[str]
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _brainda_entrypoints() -> list[str]:
    modules = [
        "metabci.brainda",
        "metabci.brainda.datasets",
        "metabci.brainda.paradigms",
        "metabci.brainda.algorithms.feature_analysis.freq_analysis",
        "metabci.brainda.algorithms.utils.model_selection",
    ]
    imported = []
    for module_name in modules:
        importlib.import_module(module_name)
        imported.append(module_name)
    return imported


def describe_public_sleep_pipeline_alignment() -> dict:
    status = inspect_metabci_components()["brainda"]
    return {
        "metabci_component": status.to_dict(),
        "project_public_sleep_modules": [
            "MetaSleepGuard.datasets.public_sleep.loaders",
            "MetaSleepGuard.preprocessing.epoching",
            "MetaSleepGuard.features",
            "MetaSleepGuard.models.public_sleep_real_baseline",
        ],
        "brainda_usage": [
            "import metabci.brainda datasets/paradigms/feature-analysis/model-selection modules",
            "use metabci.brainda.algorithms.utils.model_selection.EnhancedLeaveOneGroupOut for subject-level split checks",
            "keep Sleep-EDF/ISRUC loaders local because this MetaBCI install does not expose dedicated sleep-staging datasets",
        ],
        "project_additions": [
            "30-second sleep-window integrity checks",
            "sleep quality audit and trusted abstention",
            "public Sleep-EDF small-sample subject-level RandomForest baseline",
            "automatic HTML/Markdown/CSV/JSON reporting",
        ],
    }


def build_subject_level_splits(labels: Sequence[str], subjects: Sequence[str]) -> dict:
    status = inspect_metabci_components()["brainda"]
    if not status.available:
        return {
            "ok": False,
            "split_method": "",
            "n_splits": 0,
            "held_out_subjects": [],
            "no_subject_overlap": False,
            "error": status.error or status.detail,
        }
    if len(labels) != len(subjects):
        raise ValueError("labels and subjects must have the same length")
    if len(set(subjects)) < 2:
        raise ValueError("at least two subjects are required for subject-level splits")

    model_selection = importlib.import_module("metabci.brainda.algorithms.utils.model_selection")
    splitter = model_selection.EnhancedLeaveOneGroupOut(return_validate=False)
    x = np.zeros((len(labels), 1), dtype=float)
    y = np.asarray(labels)
    groups = np.asarray(subjects)
    split_rows = []
    no_overlap = True
    for fold, (train_index, test_index) in enumerate(splitter.split(x, y, groups=groups), start=1):
        train_subjects = set(groups[train_index].tolist())
        test_subjects = set(groups[test_index].tolist())
        overlap = sorted(train_subjects & test_subjects)
        no_overlap = no_overlap and not overlap
        split_rows.append(
            {
                "fold": fold,
                "train_size": int(len(train_index)),
                "test_size": int(len(test_index)),
                "held_out_subjects": sorted(test_subjects),
                "subject_overlap": overlap,
            }
        )
    return {
        "ok": bool(split_rows) and no_overlap,
        "split_method": "metabci.brainda.algorithms.utils.model_selection.EnhancedLeaveOneGroupOut",
        "n_splits": len(split_rows),
        "held_out_subjects": [row["held_out_subjects"][0] for row in split_rows if row["held_out_subjects"]],
        "no_subject_overlap": no_overlap,
        "splits": split_rows,
    }


def run_brainda_adapter_smoke() -> BraindaAdapterResult:
    status = inspect_metabci_components()["brainda"]
    if not status.available:
        return BraindaAdapterResult(
            ok=False,
            metabci_brainda_available=False,
            split_method="",
            n_splits=0,
            held_out_subjects=[],
            no_subject_overlap=False,
            entrypoints=[],
            notes=[status.error or status.detail],
        )

    entrypoints = _brainda_entrypoints()
    split_summary = build_subject_level_splits(
        labels=["W", "N1", "N2", "W", "N1", "N2"],
        subjects=["S1", "S1", "S2", "S2", "S3", "S3"],
    )
    return BraindaAdapterResult(
        ok=bool(split_summary["ok"]),
        metabci_brainda_available=True,
        split_method=split_summary["split_method"],
        n_splits=int(split_summary["n_splits"]),
        held_out_subjects=list(split_summary["held_out_subjects"]),
        no_subject_overlap=bool(split_summary["no_subject_overlap"]),
        entrypoints=entrypoints,
        notes=[
            "MetaBCI Brainda was imported from the local metabci environment.",
            "EnhancedLeaveOneGroupOut was exercised for subject-level split validation.",
            "Project Sleep-EDF/ISRUC loaders remain local and are documented as Brainda-aligned public-data adapters.",
        ],
    )
