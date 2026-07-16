"""Stable public wrapper around MetaSleep-Guard's trusted rejection."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from MetaSleepGuard.rejection.rejector import (
    ActiveRejector,
    RejectionDecision,
    apply_rejection,
)


class TrustedRejector:
    """Return a stage only when model confidence and signal quality permit it."""

    def __init__(
        self,
        confidence_threshold: float = 0.55,
        uncertain_quality_threshold: float = 0.70,
    ) -> None:
        self._rejector = ActiveRejector(
            confidence_threshold=confidence_threshold,
            uncertain_quality_threshold=uncertain_quality_threshold,
        )

    @property
    def confidence_threshold(self) -> float:
        return self._rejector.confidence_threshold

    @property
    def uncertain_quality_threshold(self) -> float:
        return self._rejector.uncertain_quality_threshold

    def decide(
        self,
        probabilities: Sequence[float],
        classes: Sequence[str],
        quality_grade: str | None = None,
        is_reliable: bool = True,
    ) -> RejectionDecision:
        return self._rejector.decide(
            probabilities,
            classes,
            quality_grade=quality_grade,
            is_reliable=is_reliable,
        )

    def decide_batch(
        self,
        probabilities: np.ndarray,
        classes: Sequence[str],
        quality_grades: Sequence[str] | None = None,
        reliability: Sequence[bool] | None = None,
    ) -> list[RejectionDecision]:
        return apply_rejection(
            probabilities,
            classes,
            quality_grades=quality_grades,
            reliability=reliability,
            rejector=self._rejector,
        )
