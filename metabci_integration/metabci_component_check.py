"""Runtime checks for locally importable MetaBCI framework components."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import pkgutil
from pathlib import Path
import sys
from types import ModuleType


@dataclass(frozen=True)
class ComponentStatus:
    component: str
    module: str
    available: bool
    role: str
    import_path: str | None = None
    error: str | None = None
    detail: str = ""
    package_path: str | None = None
    checked_modules: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["checked_modules"] = list(self.checked_modules)
        return payload


CORE_ROLE = "MetaBCI core package imported by the application integration layer"
COMPONENTS = {
    "brainflow": (
        "metabci.brainflow",
        "MetaBCI online acquisition, worker, marker, and ring-buffer foundation",
    ),
    "brainstim": (
        "metabci.brainstim",
        "MetaBCI stimulation/paradigm foundation for calibration prompts and markers",
    ),
    "brainda": (
        "metabci.brainda",
        "MetaBCI dataset, paradigm, feature-analysis, and model-selection foundation",
    ),
}
COMPONENT_CHILDREN = {
    "brainflow": (
        "metabci.brainflow.amplifiers",
        "metabci.brainflow.workers",
        "metabci.brainflow.logger",
    ),
    "brainstim": (
        "metabci.brainstim.framework",
        "metabci.brainstim.paradigm",
        "metabci.brainstim.utils",
    ),
    "brainda": (
        "metabci.brainda.datasets",
        "metabci.brainda.paradigms",
        "metabci.brainda.algorithms.feature_analysis.freq_analysis",
        "metabci.brainda.algorithms.utils.model_selection",
    ),
}


def _import_module(module_name: str) -> tuple[ModuleType | None, str | None]:
    try:
        return importlib.import_module(module_name), None
    except Exception as exc:  # pragma: no cover - exact optional dependency differs by env
        return None, f"{type(exc).__name__}: {exc}"


def _module_path(module: ModuleType | None) -> str | None:
    if module is None:
        return None
    path = getattr(module, "__file__", None)
    return str(path) if path else None


def _candidate_package_path(module_name: str) -> str | None:
    if not module_name.startswith("metabci."):
        return None
    root, error = _import_module("metabci")
    if root is None or error:
        return None
    root_file = getattr(root, "__file__", None)
    if not root_file:
        return None
    root_dir = Path(root_file).resolve().parent
    parts = module_name.split(".")[1:]
    package_dir = root_dir.joinpath(*parts)
    if package_dir.exists():
        return str(package_dir)
    module_file = package_dir.with_suffix(".py")
    if module_file.exists():
        return str(module_file)
    return None


def _probe_component(component: str, module_name: str, role: str) -> ComponentStatus:
    module, error = _import_module(module_name)
    checked_modules: list[str] = []
    if module is not None:
        for child in COMPONENT_CHILDREN.get(component, ()):
            child_module, child_error = _import_module(child)
            if child_module is not None and not child_error:
                checked_modules.append(child)
        return ComponentStatus(
            component=component,
            module=module_name,
            available=True,
            role=role,
            import_path=_module_path(module),
            detail="imported successfully",
            package_path=_candidate_package_path(module_name),
            checked_modules=tuple(checked_modules),
        )
    package_path = _candidate_package_path(module_name)
    detail = "module is not importable in this Python environment"
    if package_path:
        detail = "package files were found, but import failed in this Python environment"
    return ComponentStatus(
        component=component,
        module=module_name,
        available=False,
        role=role,
        error=error,
        detail=detail,
        package_path=package_path,
        checked_modules=tuple(checked_modules),
    )


def inspect_metabci_core() -> ComponentStatus:
    module, error = _import_module("metabci")
    if module is None:
        return ComponentStatus(
            component="metabci",
            module="metabci",
            available=False,
            role=CORE_ROLE,
            error=error,
            detail="MetaBCI root package is not importable",
        )
    return ComponentStatus(
        component="metabci",
        module="metabci",
        available=True,
        role=CORE_ROLE,
        import_path=_module_path(module),
        detail="imported successfully",
        package_path=str(Path(module.__file__).resolve().parent) if getattr(module, "__file__", None) else None,
    )


def inspect_metabci_components() -> dict[str, ComponentStatus]:
    return {
        component: _probe_component(component, module, role)
        for component, (module, role) in COMPONENTS.items()
    }


def inspect_metabci_framework() -> dict[str, ComponentStatus]:
    return {"metabci": inspect_metabci_core(), **inspect_metabci_components()}


def discover_metabci_module_tree(max_modules: int = 200) -> dict:
    payload = {
        "python": sys.executable,
        "metabci_imported": False,
        "metabci_file": None,
        "importable_modules": [],
        "failed_modules": {},
        "truncated_after": None,
    }
    metabci, error = _import_module("metabci")
    if metabci is None:
        payload["metabci_error"] = error
        return payload
    payload["metabci_imported"] = True
    payload["metabci_file"] = _module_path(metabci)
    package_paths = getattr(metabci, "__path__", None)
    if package_paths is None:
        return payload

    for index, modinfo in enumerate(pkgutil.walk_packages(package_paths, "metabci.")):
        if index >= max_modules:
            payload["truncated_after"] = max_modules
            break
        module, module_error = _import_module(modinfo.name)
        if module is None:
            payload["failed_modules"][modinfo.name] = module_error
            continue
        payload["importable_modules"].append(
            {
                "name": modinfo.name,
                "is_package": bool(modinfo.ispkg),
                "file": _module_path(module),
            }
        )
    return payload


def format_component_statuses(statuses: dict[str, ComponentStatus]) -> str:
    lines = []
    for name, status in statuses.items():
        outcome = "PASS" if status.available else "UNAVAILABLE"
        suffix = f" path={status.import_path}" if status.import_path else ""
        if status.error:
            suffix += f" error={status.error}"
        if status.package_path and not status.import_path:
            suffix += f" package_path={status.package_path}"
        lines.append(f"{name}: {outcome} module={status.module}{suffix} role={status.role}")
        if status.checked_modules:
            lines.append("  checked_modules=" + ", ".join(status.checked_modules))
        if status.detail:
            lines.append(f"  detail={status.detail}")
    return "\n".join(lines)


def format_module_tree(tree: dict, limit: int = 120) -> str:
    lines = [
        f"python={tree.get('python')}",
        f"metabci_imported={tree.get('metabci_imported')} file={tree.get('metabci_file')}",
        "metabci importable module tree:",
    ]
    modules = tree.get("importable_modules", [])
    for row in modules[:limit]:
        marker = "[pkg]" if row.get("is_package") else "[mod]"
        lines.append(f"  {marker} {row.get('name')} -> {row.get('file')}")
    if len(modules) > limit:
        lines.append(f"  ... {len(modules) - limit} additional importable modules omitted")
    failed = tree.get("failed_modules", {})
    if failed:
        lines.append("metabci modules found but not importable:")
        for name, error in sorted(failed.items()):
            lines.append(f"  [fail] {name}: {error}")
    if tree.get("truncated_after"):
        lines.append(f"module tree truncated_after={tree['truncated_after']}")
    if tree.get("metabci_error"):
        lines.append(f"metabci_error={tree['metabci_error']}")
    return "\n".join(lines)
