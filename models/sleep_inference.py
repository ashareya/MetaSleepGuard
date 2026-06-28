"""Online sleep-stage inference for one 30-second epoch at a time."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Sequence

import numpy as np

from ..features import extract_epoch_features
from ..quality.quality_audit import QualityResult, audit_epoch
from ..rejection.rejector import ActiveRejector, RejectionDecision
from .train_xgb import load_model_bundle, predict_class_probabilities


class SleepInference:
    """Inference wrapper with causal feature history and active rejection."""

    def __init__(self, model_bundle: dict | str | Path, rejector: ActiveRejector | None = None) -> None:
        self.bundle = load_model_bundle(model_bundle) if isinstance(model_bundle, (str, Path)) else model_bundle
        self.model = self.bundle["model"]
        self.classes = list(self.bundle["classes"])
        self.base_feature_names = list(self.bundle["base_feature_names"])
        self.context_history = int(self.bundle.get("metadata", {}).get("context_history", 2))
        self.history: deque[np.ndarray] = deque(maxlen=self.context_history)
        self.rejector = rejector or ActiveRejector()

    def reset(self) -> None:
        self.history.clear()

    def predict_epoch(
        self,
        epoch: np.ndarray,
        sfreq: float,
        channel_names: Sequence[str],
        quality: QualityResult | None = None,
    ) -> dict:
        quality = quality or audit_epoch(epoch, sfreq, channel_names)
        base = self._base_vector(epoch, sfreq, channel_names)
        context_parts = []
        missing = self.context_history - len(self.history)
        for _ in range(missing):
            context_parts.append(np.zeros_like(base))
        context_parts.extend(list(self.history))
        context_parts.append(base)
        X = np.concatenate(context_parts).reshape(1, -1)
        probabilities = predict_class_probabilities(self.model, X, len(self.classes))[0]
        decision = self.rejector.decide(probabilities, self.classes, quality.quality_grade, quality.is_reliable)
        self.history.append(base if quality.is_reliable else np.zeros_like(base))
        return {
            "stage": decision.stage,
            "confidence": decision.confidence,
            "accepted": decision.accepted,
            "reason": decision.reason,
            "probabilities": {label: float(probabilities[i]) for i, label in enumerate(self.classes)},
            "quality": quality.to_row(),
        }

    def _base_vector(self, epoch: np.ndarray, sfreq: float, channel_names: Sequence[str]) -> np.ndarray:
        canonical_names = [f"EEG{i + 1}" for i in range(np.asarray(epoch).shape[0])]
        features = extract_epoch_features(epoch, sfreq, canonical_names)
        return np.array([features.get(name, 0.0) for name in self.base_feature_names], dtype=float)


def decision_to_row(decision: RejectionDecision) -> dict:
    return {
        "stage": decision.stage,
        "confidence": decision.confidence,
        "accepted": decision.accepted,
        "reason": decision.reason,
        "quality_grade": decision.quality_grade,
    }
