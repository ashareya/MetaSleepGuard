"""Minimal dashboard generator for file replay outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def write_static_dashboard(rows: Sequence[dict], figures: Sequence[str], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_html = "\n".join(f'<img src="{Path(fig).name}" alt="{Path(fig).stem}" />' for fig in figures)
    table_rows = "\n".join(
        "<tr>"
        + "".join(f"<td>{row.get(key, '')}</td>" for key in ["window_start_time", "stage", "confidence", "quality_grade", "reason"])
        + "</tr>"
        for row in rows
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="5">
<title>MetaSleep-Guard Dashboard</title>
<style>
body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 24px; color: #1f2933; }}
img {{ max-width: 100%; display: block; margin: 16px 0; border: 1px solid #d7dde5; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border: 1px solid #d7dde5; padding: 6px 8px; font-size: 13px; }}
</style>
</head>
<body>
<h1>眠卫 MetaSleep-Guard 运行面板</h1>
{figure_html}
<table>
<thead><tr><th>Start</th><th>Stage</th><th>Confidence</th><th>Quality</th><th>Reason</th></tr></thead>
<tbody>{table_rows}</tbody>
</table>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")
    return output_path
