"""Report availability of MetaBCI core, Brainda, BrainFlow, and Brainstim."""

from __future__ import annotations

from MetaSleepGuard.experiments.common import output_dir, write_json
from MetaSleepGuard.metabci_integration import (
    discover_metabci_module_tree,
    format_component_statuses,
    format_module_tree,
    inspect_metabci_framework,
)


def main() -> None:
    statuses = inspect_metabci_framework()
    module_tree = discover_metabci_module_tree(max_modules=200)
    payload = {
        "framework": {name: status.to_dict() for name, status in statuses.items()},
        "module_tree": module_tree,
    }
    path = write_json(payload, output_dir() / "metabci_component_status.json")
    print("MetaBCI core framework status")
    print(format_component_statuses(statuses))
    print(format_module_tree(module_tree, limit=120))
    print(f"status_json={path}")


if __name__ == "__main__":
    main()
