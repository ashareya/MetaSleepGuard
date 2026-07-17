"""HTML report generator for public-data and OpenBCI runs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Sequence


LIMITATION_TEXT = (
    "公开 Sleep-EDF/ISRUC 数据用于带专家标签的分期准确率验证;"
    "博睿康 BDF/FIF 与 OpenBCI 数据用于真实文件兼容、实时链路、质量审计、可信拒识和报告生成验证;"
    "无 PSG 或专家 30 秒标签时,不声称真实睡眠五分类准确率或临床准确率。"
)


def generate_html_report(
    output_path: str | Path,
    experiment_info: dict,
    window_rows: Sequence[dict],
    figures: Sequence[str] | None = None,
    limitation_text: str = LIMITATION_TEXT,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figures = list(figures or [])
    grade_counts = Counter(str(row.get("quality_grade", "")) for row in window_rows)
    stage_counts = Counter(str(row.get("stage", "")) for row in window_rows)
    rejected = sum(1 for row in window_rows if str(row.get("stage")) == "暂不判定" or not bool(row.get("accepted", True)))
    confidence = [float(row.get("confidence", 0.0) or 0.0) for row in window_rows]
    rows_html = "\n".join(
        "<tr>"
        + "".join(
            f"<td>{row.get(key, '')}</td>"
            for key in [
                "window_start_time",
                "window_end_time",
                "stage",
                "confidence",
                "quality_grade",
                "bad_flags",
                "reason",
            ]
        )
        + "</tr>"
        for row in window_rows
    )
    info_html = "\n".join(f"<tr><th>{key}</th><td>{value}</td></tr>" for key, value in experiment_info.items())
    figure_html = "\n".join(f'<figure><img src="{Path(fig).name}" alt="{Path(fig).stem}"><figcaption>{Path(fig).stem}</figcaption></figure>' for fig in figures)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>MetaSleep-Guard Report</title>
<style>
body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 28px; color: #1f2933; line-height: 1.55; }}
h1, h2 {{ color: #16324f; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
th, td {{ border: 1px solid #d7dde5; padding: 7px 9px; font-size: 13px; text-align: left; }}
th {{ background: #eef3f8; }}
img {{ max-width: 100%; border: 1px solid #d7dde5; }}
.note {{ background: #fff8e6; border-left: 4px solid #bf7f00; padding: 10px 14px; }}
</style>
</head>
<body>
<h1>眠卫 MetaSleep-Guard 自动报告</h1>
<h2>实验基本信息、设备与通道</h2>
<table>{info_html}</table>
<h2>系统运行统计</h2>
<table>
<tr><th>30 秒窗口数量</th><td>{len(window_rows)}</td></tr>
<tr><th>睡眠阶段分布</th><td>{dict(stage_counts)}</td></tr>
<tr><th>信号质量等级分布</th><td>{dict(grade_counts)}</td></tr>
<tr><th>暂不判定窗口数</th><td>{rejected}</td></tr>
<tr><th>暂不判定比例</th><td>{(rejected / len(window_rows)) if window_rows else 0:.3f}</td></tr>
<tr><th>平均可信度</th><td>{(sum(confidence) / len(confidence)) if confidence else 0:.3f}</td></tr>
</table>
<h2>趋势与代表性图</h2>
{figure_html}
<h2>异常、拒识与窗口明细</h2>
<table>
<thead><tr><th>开始</th><th>结束</th><th>阶段</th><th>可信度</th><th>质量</th><th>异常</th><th>原因</th></tr></thead>
<tbody>{rows_html}</tbody>
</table>
<h2>限制说明</h2>
<p class="note">{limitation_text}</p>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")
    return output_path

