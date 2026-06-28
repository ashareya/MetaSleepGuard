"""Trusted active rejection logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class RejectionDecision:
    stage: str
    confidence: float
    accepted: bool
    reason: str
    quality_grade: str | None = None


class ActiveRejector:
    """Turn calibrated probabilities and quality results into trusted output."""

    def __init__(self, confidence_threshold: float = 0.55, uncertain_quality_threshold: float = 0.70) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.uncertain_quality_threshold = float(uncertain_quality_threshold)

    def decide(
        self,
        probabilities: Sequence[float],
        classes: Sequence[str],
        quality_grade: str | None = None,
        is_reliable: bool = True,
    ) -> RejectionDecision:
        probs = np.asarray(probabilities, dtype=float)
        if probs.ndim != 1:
            raise ValueError("probabilities must be one-dimensional")
        if probs.size == 0 or probs.size != len(classes):
            raise ValueError("probabilities and classes must have the same non-zero length")
        if not np.isfinite(probs).all():
            raise ValueError("probabilities must be finite")
        max_index = int(np.argmax(probs))
        confidence = float(probs[max_index])
        stage = list(classes)[max_index]
        if quality_grade == "D" or not is_reliable:
            return RejectionDecision("暂不判定", confidence, False, "low_signal_quality", quality_grade)
        if confidence < self.confidence_threshold:
            return RejectionDecision("暂不判定", confidence, False, "low_model_confidence", quality_grade)
        if quality_grade == "C" and confidence < self.uncertain_quality_threshold:
            return RejectionDecision("暂不判定", confidence, False, "quality_and_model_uncertain", quality_grade)
        return RejectionDecision(stage, confidence, True, "accepted", quality_grade)


def apply_rejection(
    probabilities: np.ndarray,
    classes: Sequence[str],
    quality_grades: Sequence[str] | None = None,
    reliability: Sequence[bool] | None = None,
    rejector: ActiveRejector | None = None,
) -> list[RejectionDecision]:
    rejector = rejector or ActiveRejector()
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (epochs, classes)")
    n = probabilities.shape[0]
    quality_grades = [None] * n if quality_grades is None else list(quality_grades)
    reliability = [True] * n if reliability is None else list(reliability)
    if len(quality_grades) != n or len(reliability) != n:
        raise ValueError("quality arrays must match the probability row count")
    return [
        rejector.decide(probabilities[i], classes, quality_grades[i], bool(reliability[i]))
        for i in range(n)
    ]
