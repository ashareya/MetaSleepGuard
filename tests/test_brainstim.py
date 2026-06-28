import csv

from MetaSleepGuard.brainstim_task.calibration_task import TaskEvent, run_calibration_task


def test_brainstim_log_has_lsl_and_wall_timestamps(tmp_path):
    path = run_calibration_task(
        tmp_path / "markers.csv",
        events=[TaskEvent("blink", "请眨眼", 0)],
        countdown_sec=0,
        dry_run=True,
        use_psychopy=False,
    )
    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["marker"] == "blink"
    assert float(row["lsl_timestamp"]) > 0
    assert float(row["unix_timestamp"]) > 0
