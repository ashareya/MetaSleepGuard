from MetaSleepGuard.reports.report_generator import generate_html_report


def test_report_generation(tmp_path):
    path = generate_html_report(
        tmp_path / "report.html",
        {"mode": "test", "device": "synthetic"},
        [
            {
                "window_start_time": 0,
                "window_end_time": 30,
                "stage": "W",
                "confidence": 0.8,
                "accepted": True,
                "quality_grade": "A",
                "bad_flags": "",
                "reason": "accepted",
            }
        ],
    )
    text = path.read_text(encoding="utf-8")
    assert "MetaSleep-Guard" in text
    assert "无 PSG" in text

