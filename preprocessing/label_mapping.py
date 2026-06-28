"""Sleep stage normalization and task mappings."""

from __future__ import annotations

import re
from typing import Iterable

CANONICAL_STAGES = ["W", "N1", "N2", "N3", "REM"]
TASK_CLASSES = {
    "5class": ["W", "N1", "N2", "N3", "REM"],
    "4class": ["W", "LIGHT", "N3", "REM"],
    "3class": ["W", "NREM", "REM"],
}


def normalize_task(task: str | int) -> str:
    text = str(task).lower().replace("_", "").replace("-", "")
    if text in {"5", "5class", "five", "fiveclass"}:
        return "5class"
    if text in {"4", "4class", "four", "fourclass"}:
        return "4class"
    if text in {"3", "3class", "three", "threeclass"}:
        return "3class"
    raise ValueError(f"unsupported sleep staging task: {task}")


def map_raw_stage(raw_stage: object) -> str:
    """Map EDF/ISRUC/raw labels to W/N1/N2/N3/REM/UNKNOWN."""

    text = str(raw_stage).strip().upper()
    text = text.replace("SLEEP STAGE", "").replace("STAGE", "").strip()
    text = re.sub(r"[^A-Z0-9]+", "", text)
    if text in {"W", "WAKE", "0"}:
        return "W"
    if text in {"N1", "S1", "1"}:
        return "N1"
    if text in {"N2", "S2", "2"}:
        return "N2"
    if text in {"N3", "S3", "3", "N4", "S4", "4"}:
        return "N3"
    if text in {"R", "REM", "5"}:
        return "REM"
    return "UNKNOWN"


def map_stage_for_task(stage: str, task: str | int = "5class") -> str:
    canonical = map_raw_stage(stage)
    task_name = normalize_task(task)
    if canonical == "UNKNOWN":
        return "UNKNOWN"
    if task_name == "5class":
        return canonical
    if task_name == "4class":
        if canonical in {"N1", "N2"}:
            return "LIGHT"
        return canonical
    if task_name == "3class":
        if canonical in {"N1", "N2", "N3"}:
            return "NREM"
        return canonical
    raise AssertionError("unreachable")


def task_classes(task: str | int) -> list[str]:
    return TASK_CLASSES[normalize_task(task)].copy()


def encode_labels(labels: Iterable[str], task: str | int = "5class") -> tuple[list[int], list[str], list[bool]]:
    classes = task_classes(task)
    index = {label: i for i, label in enumerate(classes)}
    encoded: list[int] = []
    keep: list[bool] = []
    for label in labels:
        mapped = map_stage_for_task(label, task)
        keep.append(mapped in index)
        encoded.append(index.get(mapped, -1))
    return encoded, classes, keep


def decode_labels(indices: Iterable[int], classes: Iterable[str]) -> list[str]:
    classes_list = list(classes)
    return [classes_list[int(i)] if 0 <= int(i) < len(classes_list) else "UNKNOWN" for i in indices]

