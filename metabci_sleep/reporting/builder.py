"""HTML, Markdown, CSV, and JSON report builder."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

from MetaSleepGuard.reports.report_generator import LIMITATION_TEXT, generate_html_report

from metabci_sleep.algorithms.metrics import SleepMetrics


class SleepReportBuilder:
    def __init__(self, output_dir: str | Path, epoch_sec: float = 30.0) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = SleepMetrics(epoch_sec=epoch_sec)

    def build(
        self,
        window_rows: Sequence[dict],
        experiment_info: dict | None = None,
        figures: Sequence[str] | None = None,
    ) -> dict:
        rows = list(window_rows)
        stages = [str(row.get("stage", "UNKNOWN")) for row in rows]
        structure = self.metrics.compute(stages)
        info = dict(experiment_info or {})
        info["evidence_boundary"] = LIMITATION_TEXT
        html = generate_html_report(self.output_dir / "report.html", info, rows, figures=figures)
        json_path = self.output_dir / "summary.json"
        json_path.write_text(
            json.dumps({"experiment_info": info, "sleep_metrics": structure, "windows": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        csv_path = self.output_dir / "windows.csv"
        fields = sorted({key for row in rows for key in row})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        md_path = self.output_dir / "report.md"
        md_path.write_text(
            "# metabci_sleep report\n\n"
            + f"- Engineering sleep score: {structure['engineering_sleep_score']}\n"
            + f"- Sleep efficiency: {structure['sleep_efficiency']:.3f}\n"
            + f"- WASO minutes: {structure['waso_minutes']:.1f}\n\n"
            + f"> {structure['score_disclaimer']}\n\n"
            + f"> {LIMITATION_TEXT}\n",
            encoding="utf-8",
        )
        return {"html": str(html), "markdown": str(md_path), "csv": str(csv_path), "json": str(json_path), "metrics": structure}
