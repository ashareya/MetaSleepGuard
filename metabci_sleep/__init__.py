"""MetaBCI-compatible sleep monitoring extensions.

This package is maintained by MetaSleep-Guard and is not part of the official
MetaBCI distribution.
"""

from .algorithms import (
    CoverageRiskEvaluator,
    ProbabilityCalibrator,
    SleepFeatureExtractor,
    SleepMetrics,
    SleepQualityAuditor,
    SleepStagingEstimator,
    TrustedRejector,
)
from .datasets import ISRUCSleep, SleepEDF
from .paradigms import SleepStaging

__version__ = "0.2.0"

__all__ = [
    "SleepEDF",
    "ISRUCSleep",
    "SleepStaging",
    "SleepFeatureExtractor",
    "SleepStagingEstimator",
    "ProbabilityCalibrator",
    "CoverageRiskEvaluator",
    "SleepQualityAuditor",
    "TrustedRejector",
    "SleepMetrics",
]
