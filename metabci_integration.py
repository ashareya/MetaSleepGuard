"""Runtime discovery for the three MetaBCI integration pillars."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import find_spec


@dataclass(frozen=True)
class ComponentStatus:
    component: str
    module: str
    available: bool
    role: str

    def to_dict(self) -> dict:
        return asdict(self)


COMPONENTS = {
    "brainda": ("metabci.brainda", "public-data paradigms and dataset compatibility"),
    "brainflow": ("metabci.brainflow", "online acquisition and processing integration"),
    "brainstim": ("metabci.brainstim", "calibration and stimulation task integration"),
}


def inspect_metabci_components() -> dict[str, ComponentStatus]:
    statuses: dict[str, ComponentStatus] = {}
    for component, (module, role) in COMPONENTS.items():
        try:
            available = find_spec("metabci") is not None and find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, AttributeError):
            available = False
        statuses[component] = ComponentStatus(component, module, available, role)
    return statuses
