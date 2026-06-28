from MetaSleepGuard.preprocessing.label_mapping import encode_labels, map_raw_stage, task_classes


def test_public_stage_normalization():
    raw = ["Sleep stage W", "Sleep stage 1", "S2", "N3", "S4", "R", "Movement time"]
    assert [map_raw_stage(stage) for stage in raw] == ["W", "N1", "N2", "N3", "N3", "REM", "UNKNOWN"]


def test_three_four_five_class_tasks():
    labels = ["W", "N1", "N2", "N3", "REM"]
    expected = {
        "5class": [0, 1, 2, 3, 4],
        "4class": [0, 1, 1, 2, 3],
        "3class": [0, 1, 1, 1, 2],
    }
    for task, encoded_expected in expected.items():
        encoded, classes, keep = encode_labels(labels, task)
        assert classes == task_classes(task)
        assert encoded == encoded_expected
        assert all(keep)
