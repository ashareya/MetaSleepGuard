"""Quality audit summary helpers."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from .quality_audit import QualityResult


def summarize_quality(results: Sequence[QualityResult]) -> dict:
    grade_counts = Counter(result.quality_grade for result in results)
    flag_counts = Counter(flag for result in results for flag in result.bad_flags)
    reliable = sum(1 for result in results if result.is_reliable)
    total = len(results)
    return {
        "total_windows": total,
        "reliable_windows": reliable,
        "unreliable_windows": total - reliable,
        "reliable_ratio": reliable / total if total else 0.0,
        "grade_counts": dict(grade_counts),
        "flag_counts": dict(flag_counts),
    }

