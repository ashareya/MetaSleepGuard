"""Run all repository tests without requiring pytest."""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
import traceback
from datetime import datetime, timezone


def main() -> None:
    tests_root = Path(__file__).resolve().parent
    repo_root = tests_root.parents[1]
    temp_root = repo_root / "outputs" / "test_harness"
    temp_root.mkdir(parents=True, exist_ok=True)
    passed = 0
    failures: list[str] = []
    for path in sorted(tests_root.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(f"_metasleepguard_{path.stem}", path)
        if spec is None or spec.loader is None:
            failures.append(f"{path.name}: unable to load module")
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name, function in sorted(inspect.getmembers(module, inspect.isfunction)):
            if not name.startswith("test_"):
                continue
            test_id = f"{path.name}::{name}"
            try:
                parameters = list(inspect.signature(function).parameters)
                if not parameters:
                    function()
                elif parameters == ["tmp_path"]:
                    with tempfile.TemporaryDirectory(dir=temp_root) as temp_dir:
                        function(Path(temp_dir))
                else:
                    raise RuntimeError(f"unsupported fixtures: {parameters}")
                passed += 1
                print(f"PASS {test_id}")
            except Exception:
                failures.append(test_id)
                print(f"FAIL {test_id}")
                traceback.print_exc()
    print(f"test_result passed={passed} failed={len(failures)}")
    result_path = repo_root / "outputs" / "metasleepguard_outputs" / "test_results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "passed": passed,
                "failed": len(failures),
                "failures": failures,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"test_results_json={result_path}")
    if failures:
        raise SystemExit("Failed tests:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
