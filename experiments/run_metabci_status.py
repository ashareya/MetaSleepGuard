"""Report availability of MetaBCI Brainda, BrainFlow, and Brainstim."""

from __future__ import annotations

from MetaSleepGuard.experiments.common import output_dir, write_json
from MetaSleepGuard.metabci_integration import inspect_metabci_components


def main() -> None:
    statuses = inspect_metabci_components()
    payload = {name: status.to_dict() for name, status in statuses.items()}
    path = write_json(payload, output_dir() / "metabci_component_status.json")
    for name, status in statuses.items():
        print(f"{name}: available={status.available} module={status.module} role={status.role}")
    print(f"status_json={path}")


if __name__ == "__main__":
    main()
