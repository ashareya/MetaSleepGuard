"""Optional MetaBCI Brainda compatibility helpers.

Sleep-EDF and ISRUC are loaded by the project adapters because MetaBCI does not
guarantee those datasets in every installation. This bridge keeps Brainda as an
explicit integration boundary without making it a hard dependency for replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec


@dataclass(frozen=True)
class BraindaStatus:
    available: bool
    module: str
    detail: str


def brainda_status() -> BraindaStatus:
    """Return whether the MetaBCI Brainda package is importable."""

    module = "metabci.brainda"
    available = find_spec("metabci") is not None and find_spec(module) is not None
    detail = "Brainda integration is available" if available else "Install MetaBCI to enable Brainda datasets/paradigms"
    return BraindaStatus(available=available, module=module, detail=detail)


def require_brainda():
    """Import and return ``metabci.brainda`` or raise an actionable error."""

    status = brainda_status()
    if not status.available:
        raise RuntimeError(status.detail)
    import metabci.brainda as brainda

    return brainda
