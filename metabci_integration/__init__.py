"""MetaBCI core-framework integration layer for MetaSleep-Guard."""

from __future__ import annotations

from .metabci_brainda_adapter import (
    build_subject_level_splits,
    describe_public_sleep_pipeline_alignment,
    run_brainda_adapter_smoke,
)
from .metabci_brainflow_adapter import (
    describe_openbci_runtime_alignment,
    run_brainflow_adapter_smoke,
    verify_metabci_ring_buffer,
)
from .metabci_brainstim_adapter import (
    build_brainstim_marker_plan,
    probe_metabci_brainstim,
    run_brainstim_adapter_smoke,
)
from .metabci_component_check import (
    ComponentStatus,
    discover_metabci_module_tree,
    format_component_statuses,
    format_module_tree,
    inspect_metabci_components,
    inspect_metabci_core,
    inspect_metabci_framework,
)

__all__ = [
    "ComponentStatus",
    "build_brainstim_marker_plan",
    "build_subject_level_splits",
    "describe_openbci_runtime_alignment",
    "describe_public_sleep_pipeline_alignment",
    "discover_metabci_module_tree",
    "format_component_statuses",
    "format_module_tree",
    "inspect_metabci_components",
    "inspect_metabci_core",
    "inspect_metabci_framework",
    "probe_metabci_brainstim",
    "run_brainda_adapter_smoke",
    "run_brainflow_adapter_smoke",
    "run_brainstim_adapter_smoke",
    "verify_metabci_ring_buffer",
]
