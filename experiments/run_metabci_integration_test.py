"""Run a lightweight MetaBCI core-framework integration test."""

from __future__ import annotations

from MetaSleepGuard.experiments.common import create_run_dir, write_json
from MetaSleepGuard.metabci_integration import (
    build_brainstim_marker_plan,
    describe_openbci_runtime_alignment,
    describe_public_sleep_pipeline_alignment,
    discover_metabci_module_tree,
    format_component_statuses,
    inspect_metabci_framework,
    run_brainda_adapter_smoke,
    run_brainflow_adapter_smoke,
    run_brainstim_adapter_smoke,
)


def main() -> None:
    run_dir = create_run_dir("metabci_integration_test")
    framework = inspect_metabci_framework()
    module_tree = discover_metabci_module_tree(max_modules=200)
    brainflow = run_brainflow_adapter_smoke()
    brainda = run_brainda_adapter_smoke()
    brainstim = run_brainstim_adapter_smoke(run_dir)
    payload = {
        "framework": {name: status.to_dict() for name, status in framework.items()},
        "module_tree": module_tree,
        "brainflow_adapter": brainflow.to_dict(),
        "brainstim_adapter": brainstim.to_dict(),
        "brainstim_marker_plan": build_brainstim_marker_plan(),
        "brainda_adapter": brainda.to_dict(),
        "openbci_alignment": describe_openbci_runtime_alignment(),
        "public_sleep_alignment": describe_public_sleep_pipeline_alignment(),
    }
    output = write_json(payload, run_dir / "metabci_integration_test.json")

    print("MetaBCI core framework check")
    print(format_component_statuses(framework))
    print(f"module_tree_importable_count={len(module_tree.get('importable_modules', []))}")
    print(f"module_tree_failed_count={len(module_tree.get('failed_modules', {}))}")
    print(f"brainflow_adapter_ok={brainflow.ok}")
    print(f"brainflow_ring_buffer_full={brainflow.ring_buffer_full}")
    print(f"brainda_adapter_ok={brainda.ok}")
    print(f"brainda_subject_split_method={brainda.split_method}")
    print(f"brainstim_importable={brainstim.metabci_brainstim_available}")
    print(f"brainstim_marker_log_ok={brainstim.ok}")
    print(f"integration_json={output}")

    required = {
        "metabci": framework["metabci"].available,
        "brainflow_adapter": brainflow.ok,
        "brainda_adapter": brainda.ok,
        "brainstim_marker_log": brainstim.ok,
    }
    failed = [name for name, ok in required.items() if not ok]
    if failed:
        raise SystemExit("MetaBCI integration test failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
