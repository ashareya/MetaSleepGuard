"""Build competition submission materials from verified real OpenBCI outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
from typing import Iterable


TASKS = [
    ("metabci-integration-test", "direct run", "MetaBCI core import and adapter smoke", "metabci env", "no", "no", "yes"),
    ("status", "直接运行", "MetaBCI 组件发现", "无", "否", "否", "是"),
    ("test", "直接运行", "完整代码测试；含 synthetic smoke", "无", "部分", "否", "是"),
    ("real-openbci-report", "直接运行", "两份真实 OpenBCI 报告", "真实日志目录", "否", "是", "是"),
    ("openbci-file-replay", "直接运行", "真实 TXT、事件日志、阶段/窗口表回放审计", "真实日志目录", "否", "是", "是"),
    ("submission-pack", "直接运行", "完整初赛材料", "已生成真实报告", "否", "是", "是"),
    ("metrics-export", "直接运行", "真实质量指标、manifest、窗口 88 证据", "已生成真实报告", "否", "是", "是"),
    ("demo-assets", "直接运行", "3 分钟脚本、分镜、旁白、录屏清单", "已生成真实报告", "否", "是", "是"),
    ("prepare", "可运行但默认可能回退 synthetic", "公开数据预处理", "Sleep-EDF/ISRUC 路径", "是", "否", "否"),
    ("train", "可运行但默认可能回退 synthetic", "公开数据基线训练", "公开数据；XGBoost 缺失时使用 RF smoke fallback", "是", "否", "否"),
    ("evaluate", "可运行但默认可能回退 synthetic", "公开数据评估", "公开数据/模型", "是", "否", "否"),
    ("cross", "可运行但默认可能回退 synthetic", "双向跨数据集评估", "两个公开数据集", "是", "否", "否"),
    ("public-sleep-baseline", "提供真实目录后可运行", "三/四/五分类严格真实公开数据训练", "完整 Sleep-EDF 或 ISRUC", "否", "否", "数据就绪后是"),
    ("public-sleep-eval", "提供真实目录后可运行", "三/四/五分类严格真实公开数据评估", "完整 Sleep-EDF 或 ISRUC", "否", "否", "数据就绪后是"),
    ("cross-dataset-eval", "需两个真实数据集", "三/四/五分类双向跨数据集评估", "完整 Sleep-EDF 与 ISRUC", "否", "否", "数据就绪后是"),
    ("public-sleep-real-baseline", "数据就绪后直接运行", "Sleep-EDF 双导三/五分类真实基线", "MNE；至少 15 名 Sleep-EDF 被试", "否", "否", "是"),
    ("audit", "需输入", "BDF/FIF 文件审计", "-InputPath 指向真实 BDF/FIF", "否", "否", "有数据后是"),
    ("replay", "需输入", "单文件通用回放", "-InputPath；可选模型", "否", "可", "辅助"),
    ("realtime", "需硬件", "OpenBCI Cyton BrainFlow 实时采集", "串口和设备；或 -Synthetic", "可选", "实时模式是", "现场演示"),
    ("brainstim", "需 metabci_stim 环境", "中文标定范式", "PsychoPy/pylsl；干跑用 -Synthetic", "可选", "否", "现场演示"),
    ("report", "仅演示", "旧 synthetic 占位报告", "无", "是", "否", "不作为正式证据"),
]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _task_table() -> str:
    lines = [
        "| Task | 当前状态 | 用途 | 依赖/输入 | 使用 synthetic | 使用真实 OpenBCI | 初赛文档 |",
        "|---|---|---|---|---|---|---|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in TASKS)
    return "\n".join(lines)


def _commands() -> str:
    return r"""```powershell
# 本机 PowerShell 策略要求使用 ExecutionPolicy Bypass；metabci 为分析环境
$py = "C:\Users\ZYH\anaconda3\envs\metabci\python.exe"
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 -Task status -Python $py
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 -Task metabci-integration-test -Python $py
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 -Task test -Python $py
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 -Task real-openbci-report -Python $py
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 -Task openbci-file-replay -Python $py
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 -Task metrics-export -Python $py
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 -Task submission-pack -Python $py
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 -Task demo-assets -Python $py
```"""


def _validate_window_88(windows: list[dict[str, str]]) -> dict[str, str]:
    matches = [row for row in windows if int(row["window_index"]) == 88]
    if len(matches) != 1:
        raise RuntimeError("The real 60-minute output must contain exactly one window 88")
    window = matches[0]
    if float(window["coverage_ratio"]) != 0.0:
        raise RuntimeError("Window 88 must have coverage_ratio=0 before submission materials can be generated")
    if window["quality_grade"] != "D" or window["usable_for_window_inference"] != "False":
        raise RuntimeError("Window 88 must be grade D and rejected from inference")
    if "暂不判定" not in window["trusted_output"]:
        raise RuntimeError("Window 88 must explicitly output 暂不判定")
    return window


def _runnable_tasks_markdown(public_summary: dict | None = None) -> str:
    public_status = (
        f"- 已完成 {public_summary['n_subjects']} 名被试的真实 Sleep-EDF 三/五分类基线；"
        "synthetic 指标只保留为 smoke。"
        if public_summary
        else "- 当前没有完整 Sleep-EDF/ISRUC EDF 与专家标签；已有睡眠指标 JSON 标记为 `synthetic_demo=true`，不得作为准确率证据。"
    )
    return f"""# 当前可运行任务审计

## 结论

- `metabci` conda 环境下 `brainda=True`、`brainflow=True`、`brainstim=True`（模块可发现）。
- Brainstim 图形范式应使用 `metabci_stim` 环境；分析环境 `metabci` 不安装 PsychoPy。
{public_status}
- 当前真实 OpenBCI 数据可直接用于采集链路、文件兼容、质量审计、30 秒窗口、中断检测、主动拒识和自动报告。
- 当前分析环境没有 XGBoost；旧 smoke 会使用 RandomForest fallback。正式 XGBoost 基线需先安装 `xgboost` 并配置真实公开数据。

## Task 列表

{_task_table()}

## 推荐执行命令

{_commands()}

## 公开数据任务的防误用规则

新增的 `public-sleep-baseline`、`public-sleep-eval`、`cross-dataset-eval` 均启用严格模式：未提供真实数据目录时明确报错，不会静默生成 synthetic 指标。旧 `prepare/train/evaluate/cross` 保留 smoke 能力，但不适合作为当前初赛准确率证据。
"""


def _real_conclusion(ten: dict, sixty: dict, window_88: dict[str, str]) -> str:
    return f"""# 真实数据提交结论

## 10 分钟 Fp1/Fp2 前额双导脑电质量标定实验

### 可以写入比赛材料

- 完成真实 OpenBCI Cyton Fp1/Fp2 双导采集链路验证。
- 正式数据时间覆盖率约 {ten['coverage_ratio']:.3%}，样本完整率约 {ten['sample_completeness_ratio']:.3%}。
- 有效通道为 GUI Ch2/Fp1 与 Ch7/Fp2；停用通道不参与质量评分。
- 数据可用于睁眼、闭眼、眨眼、咬牙、转头、动线等阶段质量标定。
- 数据可用于伪迹识别、质量审计和“暂不判定”验证。

### 不能写

- 不能写“Alpha 验证成功”或“O1/O2 枕区 Alpha 验证”。
- 不能写“完成睡眠分期准确率验证”。

## 60 分钟午睡 / 闭眼休息连续采集实验

### 可以写入比赛材料

- 完成 60 分钟真实场景采集流程，事件日志覆盖完整实验区间。
- 生成 {sixty['window_log_integrity']['row_count']} 个 30 秒窗口日志。
- 原始数据时间覆盖率约 {sixty['coverage_ratio']:.3%}，样本完整率约 {sixty['sample_completeness_ratio']:.3%}。
- 缺失时长约 {sixty['missing_duration_sec']:.3f} 秒，识别到 {sixty['gap_count']} 个文件级缺口。
- 系统识别数据中断并将低质量窗口标记为“暂不判定”。
- 第 88 个窗口是明确展示案例：`coverage_ratio={float(window_88['coverage_ratio']):.0f}`、质量等级 `{window_88['quality_grade']}`、输出“{window_88['trusted_output']}”。
- 可用于连续采集、30 秒滑窗、质量审计、可信拒识和自动报告验证。

### 不能写

- 不能写“60 分钟全程无丢包”。
- 不能写“被试已确认进入睡眠”。
- 不能写“PSG 级验证”“专家标签验证完成”或“OpenBCI 五分类准确率验证”。

## 数据解释边界

真实 OpenBCI 数据没有 PSG 和专家 30 秒睡眠标签，只验证工程链路与质量守护能力。当前睡眠分期 Accuracy、Macro-F1、Cohen's Kappa 和每类指标仅来自带专家标签的 Sleep-EDF；只有取得真实 ISRUC 数据并完成复测后，才可报告 ISRUC 或跨数据集泛化结果。
"""


def _public_sleep_readme(public_summary: dict | None = None) -> str:
    if public_summary:
        metrics = public_summary["metrics"]
        return f"""# Public Sleep Metrics Status

已完成真实 Sleep-EDF / Sleep Physionet 小样本基线，不再以 synthetic smoke 作为准确率证据。

- 被试数：{public_summary['n_subjects']}
- 总有效 30 秒 Epoch：{public_summary['n_epochs_total']}
- 双导：{', '.join(public_summary['channels'])}
- 划分：{public_summary['split_method']}，同一被试不跨训练/测试集合。
- 3-class：Accuracy {metrics['3class']['accuracy']:.4f}，Macro-F1 {metrics['3class']['macro_f1']:.4f}，Kappa {metrics['3class']['cohen_kappa']:.4f}。
- 5-class：Accuracy {metrics['5class']['accuracy']:.4f}，Macro-F1 {metrics['5class']['macro_f1']:.4f}，Kappa {metrics['5class']['cohen_kappa']:.4f}。

真实公开数据指标位于 `outputs/metasleepguard_outputs/public_sleep_real_baseline/`。原 synthetic 指标仍仅用于 smoke，并在 manifest 中标记 `used_for_accuracy=no`。
"""
    return r"""# Public Sleep Metrics Status

公开睡眠数据用于睡眠分期准确率验证，当前提交提供接口、流程和小样本/占位 smoke test；真实 OpenBCI 数据不作为分期准确率验证。

## 当前状态

- 工程内没有完整 Sleep-EDF/ISRUC EDF 与专家标签文件。
- 现有 `sleep-edf_*_metrics.json` 的 `metadata.synthetic_demo=true`，仅用于检查三/四/五分类、指标导出、概率校准和拒识代码是否可运行。
- 不提交 synthetic Accuracy、Macro-F1、Kappa 或混淆矩阵作为客观准确率证据。

## 数据就绪后的严格命令

```powershell
.\run.ps1 -Task public-sleep-baseline -Dataset sleep-edf -DataRoot D:\data\SleepEDF
.\run.ps1 -Task public-sleep-eval -Dataset sleep-edf -DataRoot D:\data\SleepEDF
.\run.ps1 -Task cross-dataset-eval -SleepEdfRoot D:\data\SleepEDF -IsrucRoot D:\data\ISRUC
```

这些任务禁用 synthetic fallback，并按被试划分，禁止随机打乱 Epoch 造成泄漏。
"""


def _real_items(ten: dict, sixty: dict, public_summary: dict | None = None) -> str:
    public_line = (
        f"- Sleep-EDF 真实公开基线：{public_summary['n_subjects']} 名被试、"
        f"{public_summary['n_epochs_total']} 个 30 秒 Epoch，用于分期准确率证据。"
        if public_summary
        else "- Sleep-EDF/ISRUC 真实指标尚未生成；synthetic smoke 不用于准确率证据。"
    )
    evidence_boundary = (
        "OpenBCI 数据不能证明睡眠分期准确率、PSG 级一致性、被试确认入睡、全程无丢包或 O1/O2 Alpha 验证。"
        "分期 Accuracy/Macro-F1/Kappa 由本提交中的真实 Sleep-EDF 专家标签基线提供；synthetic smoke 不作为证据。"
        if public_summary
        else "当前没有真实公开睡眠指标；OpenBCI 与 synthetic smoke 均不能作为睡眠分期准确率证据。"
    )
    return f"""# 真实数据可运行项目

## 已有真实数据

- 10 分钟 Fp1/Fp2 前额双导质量标定：时间覆盖率 {ten['coverage_ratio']:.3%}，样本完整率 {ten['sample_completeness_ratio']:.3%}。
- 60 分钟午睡/闭眼休息连续采集：120 个窗口，时间覆盖率 {sixty['coverage_ratio']:.3%}，缺失 {sixty['missing_duration_sec']:.3f} 秒。
{public_line}

## 可运行任务与输出

| 命令 | 真实数据用途 | 主要输出 |
|---|---|---|
| `run.ps1 -Task real-openbci-report` | 重新解析全部日志并生成正式报告 | `real_openbci_reports/` |
| `run.ps1 -Task openbci-file-replay` | TXT 回放、通道识别、事件对齐、阶段/窗口审计 | 同上 |
| `run.ps1 -Task metrics-export` | 导出窗口 88、低质量窗口、质量汇总及 manifest | `03_metric_validation_data/` |
| `run.ps1 -Task submission-pack` | 生成初赛文档与视频素材 | `outputs/metasleepguard_submission/` |
| `run.ps1 -Task public-sleep-real-baseline` | Sleep-EDF 双导三/五分类真实公开数据基线 | `public_sleep_real_baseline/` |

## 已验证能力

- OpenBCI GUI TXT 兼容，自动读取 250 Hz、8 个 EXG 通道。
- 仅 GUI Ch2/Fp1、Ch7/Fp2 参与双导分析；停用通道不参与评分。
- 工频、强工频、基线漂移、平线、饱和、运动伪迹、数据中断、异常幅值、零方差及有效数据比例不足审计。
- 30 秒窗口、A/B/C/D 分级、低质量拒识、“暂不判定”和自动报告。

## 不能证明

{evidence_boundary}
"""


def _objective_evidence(repo_root: Path, reports_root: Path, public_summary: dict | None = None) -> str:
    public_metrics = ""
    if public_summary:
        three = public_summary["metrics"]["3class"]
        five = public_summary["metrics"]["5class"]
        public_metrics = f"""
## Sleep-EDF 真实公开数据基线

- 功能描述：使用 Sleep-EDF / Sleep Physionet subjects {', '.join(public_summary['subject_ids'])}、Fpz-Cz/Pz-Oz 双导、30 秒 Epoch 完成三分类和五分类基线。
- 被试数：{public_summary['n_subjects']}；有效 Epoch：{public_summary['n_epochs_total']}。
- 代码路径：`experiments/run_public_sleep_real_baseline.py`、`models/public_sleep_real_baseline.py`。
- 测试命令：`run.ps1 -Task public-sleep-real-baseline -Python $py`。
- 输出路径：`outputs/metasleepguard_outputs/public_sleep_real_baseline/`。
- 3-class 指标：Accuracy {three['accuracy']:.4f}，Macro-F1 {three['macro_f1']:.4f}，Kappa {three['cohen_kappa']:.4f}。
- 5-class 指标：Accuracy {five['accuracy']:.4f}，Macro-F1 {five['macro_f1']:.4f}，Kappa {five['cohen_kappa']:.4f}。
- 可写进文档：真实公开数据、专家标签、GroupKFold 被试级划分，无 Epoch 随机泄漏。
- 边界限制：{public_summary['n_subjects']} 名被试、每人一晚的子集传统模型基线，不代表完整 Sleep-EDF 队列性能。
"""
    brainda_boundary = (
        f"已完成 {public_summary['n_subjects']} 名 Sleep-EDF 被试真实基线；ISRUC 与跨数据集真实评估仍待数据。"
        if public_summary
        else "当前没有真实 Sleep-EDF/ISRUC 文件，不能声称已获得公开数据准确率。"
    )
    dataset_output = (
        "`outputs/metasleepguard_outputs/public_sleep_real_baseline/` 与提交目录 `public_sleep_metrics/`。"
        if public_summary
        else "`public_sleep_metrics/README.md` 当前记录未具备正式数据。"
    )
    dataset_boundary = (
        "Sleep-EDF 真实小样本指标已完成；ISRUC 与真实跨数据集指标仍待数据。"
        if public_summary
        else "接口完成不等于真实数据指标完成。"
    )
    algorithm_boundary = (
        "本次可提交真实 Sleep-EDF 指标来自明确标注的 RandomForest 传统基线；XGBoost 仍是可选增强，不能把 RF 指标写成 XGBoost 指标。"
        if public_summary
        else "当前环境缺 XGBoost，synthetic smoke 使用 RF fallback；正式模型指标需补数据与依赖。"
    )
    model_evidence_boundary = (
        f"已完成 {public_summary['n_subjects']} 名真实 Sleep-EDF 被试的三/五分类准确率验证；"
        "OpenBCI 数据只用于采集与质量链路验证。"
        if public_summary
        else "模型准确率阶段仍等待真实公开睡眠数据。"
    )
    return f"""# 客观评分证据表

{public_metrics}

## Brainda 使用证据

- 功能描述：检测 `metabci.brainda`，以 Brainda 为公开数据/范式兼容边界，并提供 Sleep-EDF、ISRUC 适配器。
- 代码路径：`metabci_integration/`、`metabci_sleep/datasets/`、`datasets/public_sleep/loaders.py`。
- 测试命令：`run.ps1 -Task status`、`run.ps1 -Task test`。
- 输出产物：`{repo_root / 'outputs/metasleepguard_outputs/metabci_component_status.json'}`。
- 可写进文档：项目围绕 MetaBCI Brainda 兼容层实现公开睡眠数据接口与被试级评估流程。
- 边界限制：{brainda_boundary}

## BrainFlow 使用证据

- 功能描述：提供 OpenBCI Cyton 串口、BrainFlow SDK 采集、双导读取、环形缓存和 30 秒在线处理。
- 代码路径：`realtime/openbci_brainflow_stream.py`、`realtime/realtime_pipeline.py`、`experiments/run_openbci_realtime.py`。
- 测试命令：`run.ps1 -Task status`、`run.ps1 -Task test`；硬件现场使用 `run.ps1 -Task realtime -SerialPort COMx`。
- 输出产物：真实离线证据位于 `{reports_root}`。
- 可写进文档：完成 BrainFlow/OpenBCI 运行接口和真实 OpenBCI GUI 文件回放验证。
- 边界限制：现有 SX 文件由 OpenBCI GUI 保存，证明文件兼容与链路可用；不能反推本次文件由项目内 BrainFlow 类直接写出。

## Brainstim 使用证据

- 功能描述：中文提示、倒计时、睁眼/闭眼/眨眼/咬牙/转头/动线事件、LSL 与 CSV 日志。
- 代码路径：`brainstim_task/calibration_task.py`、`brainstim_task/lsl_marker.py`、`experiments/run_brainstim_calibration.py`。
- 测试命令：`run.ps1 -Task test`；图形范式在 `metabci_stim` 环境执行 `run.ps1 -Task brainstim`。
- 输出产物：标定日志由任务运行目录生成；真实实验阶段质量表位于 10 分钟报告目录。
- 可写进文档：完成面向质量标定的 Brainstim 风格范式与事件日志接口。
- 边界限制：`metabci` 分析环境无 PsychoPy，正式图形范式必须使用 `metabci_stim`。

## 新增大规模功能

- 功能描述：公开数据分期、双导特征、因果上下文、质量检测、校准拒识、实时采集、文件回放、自动报告形成完整链路。
- 代码路径：`datasets`、`preprocessing`、`features`、`models`、`quality`、`rejection`、`realtime`、`reports`。
- 测试命令：`run.ps1 -Task test`、`run.ps1 -Task real-openbci-report`。
- 输出产物：两份真实 HTML 报告、CSV、JSON、PNG 与源文件 SHA256 manifest。
- 可写进文档：新增从数据读取到可信输出和自动报告的系统级功能。
- 边界限制：{model_evidence_boundary}

## 新增数据集

- 功能描述：Sleep-EDF 与 ISRUC-Sleep 读取、标签统一、三/四/五分类及被试级划分。
- 代码路径：`datasets/public_sleep/loaders.py`、`preprocessing/label_mapping.py`。
- 测试命令：`run.ps1 -Task test`；数据就绪后使用严格 public-sleep 任务。
- 输出产物：{dataset_output}
- 可写进文档：已新增两个公开数据集接口和无 Epoch 泄漏的评估流程。
- 边界限制：{dataset_boundary}

## 新增范式 / 算法 / 设备

- 功能描述：质量标定范式、XGBoost 优先基线、Causal Context、概率校准、Coverage-Risk、OpenBCI Cyton 双导设备接入。
- 代码路径：`brainstim_task`、`models/train_xgb.py`、`features/causal_context.py`、`rejection`、`realtime/openbci_brainflow_stream.py`。
- 测试命令：`run.ps1 -Task test`、`run.ps1 -Task real-openbci-report`。
- 输出产物：真实质量等级、拒识窗口和 Coverage/缺口证据。
- 可写进文档：系统在无可信输入时主动输出“暂不判定”。
- 边界限制：{algorithm_boundary}

## 修复功能点

- 功能描述：修复被试泄漏风险、因果上下文未来信息风险、通道单位/选择、跨文件缺口插值风险、停用通道误参与质量评分风险。
- 代码路径：`preprocessing/epoching.py`、`features/causal_context.py`、`realtime/openbci_file_loader.py`、`realtime/real_openbci_data.py`。
- 测试命令：`run.ps1 -Task test`。
- 输出产物：第 88 窗口零覆盖、D 级、暂不判定；EXG3/4 瞬时活动仅审计不评分。
- 可写进文档：对数据中断和停用通道异常进行了可追溯修复与回归验证。
- 边界限制：质量阈值属于工程规则，尚无独立专家质量标签验证。

## 完全基于 MetaBCI 开发证据

- 功能描述：项目架构明确围绕 Brainda、BrainFlow、Brainstim 三支柱扩展，保留 MetaBCI 许可证说明。
- 代码路径：`metabci_integration/`、`metabci_sleep/`、`THIRD_PARTY_NOTICES.md`、`LICENSE`。
- 测试命令：`run.ps1 -Task status`。
- 输出产物：三组件状态 JSON、真实 OpenBCI 报告和 Brainstim 测试日志。
- 可写进文档：MetaSleep-Guard 是基于 MetaBCI 三组件边界扩展的双导可信睡眠质量守护应用。
- 边界限制：应表述为“基于 MetaBCI 开发”，不应暗示所有第三方数据、BrainFlow SDK 或项目新增代码均由 MetaBCI 原生提供。
"""


def _test_document(ten: dict, sixty: dict, test_result: dict, public_summary: dict | None = None) -> str:
    test_summary = (
        f"本次完整测试结果为 {test_result['passed']}/{test_result['passed'] + test_result['failed']} 通过。"
        if test_result
        else "请先运行 `run.ps1 -Task test` 生成可追溯测试结果。"
    )
    if public_summary:
        three = public_summary["metrics"]["3class"]
        five = public_summary["metrics"]["5class"]
        public_status = f"""- 项目类型：睡眠分期 / 被动监测。
- 类别数：3 类和 5 类。
- 导联数：2 导（Fpz-Cz、Pz-Oz）。
- 数据时长：30 秒 Epoch，共 {public_summary['n_epochs_total']} 个有效 Epoch，{public_summary['n_subjects']} 名被试。
- 3-class：Accuracy {three['accuracy']:.4f}，Macro-F1 {three['macro_f1']:.4f}，Kappa {three['cohen_kappa']:.4f}。
- 5-class：Accuracy {five['accuracy']:.4f}，Macro-F1 {five['macro_f1']:.4f}，Kappa {five['cohen_kappa']:.4f}。
- 测试示例程序：`python -m MetaSleepGuard.experiments.run_public_sleep_real_baseline`。
- 划分：{public_summary['split_method']}，无被试交叉。"""
        dataset_status = (
            f"新增 Sleep-EDF 与 ISRUC-Sleep 接口；当前真实 Sleep-EDF 基线包含 "
            f"{public_summary['n_subjects']} 名被试和 {public_summary['n_epochs_total']} 个有效 Epoch。"
        )
    else:
        public_status = "- 真实 Sleep-EDF/ISRUC 指标尚未生成，synthetic smoke 不作为准确率证据。"
        dataset_status = "新增 Sleep-EDF 与 ISRUC-Sleep 接口；当前仅接口与 synthetic smoke 完成，正式数据指标待补。"
    return f"""# 项目测试说明文档素材

仓库链接见项目 README

## 项目技术路径

MetaSleep-Guard 围绕 MetaBCI Brainda、BrainFlow、Brainstim 三部分建设。Sleep-EDF提供当前带专家标签的分期准确率证据，ISRUC提供兼容数据接口但不作为当前真实指标；OpenBCI Cyton Fp1/Fp2 双导负责真实采集、文件回放、质量审计、30 秒滑窗和拒识；Brainstim 标定范式负责质量/伪迹事件提示与日志。信号经通道选择、250 Hz、带通/陷波、30 秒 Epoch、手工特征和仅使用前两个 Epoch 的因果上下文进入基线模型，输出概率校准结果或“暂不判定”。

## 项目整体效果

真实 OpenBCI 数据已完成两类工程验证：10 分钟质量标定覆盖率 {ten['coverage_ratio']:.3%}；60 分钟流程生成 120 窗口，覆盖率 {sixty['coverage_ratio']:.3%}，识别 {sixty['gap_count']} 个缺口和 {len(sixty['rejected_window_indices'])} 个拒识窗口。报告自动输出 HTML、Markdown、JSON、CSV 和 PNG。

## 代码测试说明

{test_summary}

1. `run.ps1 -Task status`：核验 MetaBCI 三组件可发现性。
2. `run.ps1 -Task test`：运行全部测试，覆盖标签、被试划分、特征、质量、拒识、回放、Brainstim 与真实报告。
3. `run.ps1 -Task real-openbci-report`：从 SX 原始日志重建两份报告。
4. `run.ps1 -Task openbci-file-replay`：执行真实 TXT、事件和窗口日志回放审计。
5. `run.ps1 -Task metrics-export`：校验并导出窗口 88 证据和质量指标。

## 项目指标达成情况

- 已达成：真实采集文件兼容、双导选择、30 秒滑窗、9 类质量/完整性标志、A/B/C/D 分级、主动拒识、自动报告。
- 已达成代码流程：三/四/五分类、Accuracy、Macro-F1、Kappa、每类指标、混淆矩阵、ECE、Brier、Coverage-Risk、被试级划分和跨数据集入口。
{public_status}

## 基础功能点说明

通道选择、滤波、重采样、Epoch、标签统一、特征提取、模型评估、实时缓存、文件保存、可视化和报告均有独立模块与测试。

## 新增功能点说明

新增 Causal Context、质量守护、停用通道审计、数据中断检测、概率校准、可信拒识、BrainFlow Cyton、Brainstim 标定和一键提交材料导出。

## 新增数据集说明

{dataset_status} 标签统一为 W/N1/N2/N3/REM，并支持三/四/五分类流程。

## 修复功能点说明

修复 Epoch 随机泄漏风险、未来上下文泄漏风险、单位归一化、双导严格选择、跨文件缺口不插值、低覆盖主动拒识和停用通道异常误参与评分等问题。

## 数据限制

真实 OpenBCI 记录没有 PSG/专家睡眠标签，不用于证明睡眠分期准确率、确认入睡或 Alpha 验证；60 分钟记录存在真实缺口，不能写“全程无丢包”。
"""


def _video_assets(video_dir: Path, ten: dict, sixty: dict) -> list[Path]:
    script = f"""# 3 分钟演示视频脚本

| 时间 | 画面 | 讲解重点 |
|---|---|---|
| 0:00-0:15 | 终端运行 `status` | MetaBCI Brainda、BrainFlow、Brainstim 三组件状态 |
| 0:15-0:35 | 项目目录与技术流程图 | 公共睡眠数据负责准确率；真实 OpenBCI 负责质量守护 |
| 0:35-1:15 | 打开 10 分钟 HTML 报告 | Fp1/Fp2 双导、覆盖率 {ten['coverage_ratio']:.3%}、阶段波形/频谱、伪迹质量分级 |
| 1:15-2:15 | 打开 60 分钟 HTML 和窗口 CSV | 120 个窗口、覆盖率 {sixty['coverage_ratio']:.3%}、真实缺口、质量趋势 |
| 2:15-2:35 | 定位第 88 窗口 | coverage=0、D 级、暂不判定，展示中断检测与主动拒识 |
| 2:35-2:50 | 展示报告目录 | HTML、Markdown、summary、CSV、PNG、manifest 自动生成 |
| 2:50-3:00 | 展示数据边界页 | OpenBCI 不证明分期准确率；当前准确率由 15 名 Sleep-EDF 被试验证，ISRUC 仅完成接口测试 |
"""
    storyboard = """# 分镜表

| 镜头 | 屏幕操作 | 必须拍到 | 避免事项 |
|---|---|---|---|
| 1 | 运行 status | 三组件 True 与命令 | 不拍无关桌面信息 |
| 2 | 展开工程目录 | brainda/brainflow/brainstim 对应代码 | 不长时间滚动源码 |
| 3 | 10 分钟报告顶部与图表 | 真实数据、Fp1/Fp2、覆盖率、阶段质量 | 不说 Alpha 成功 |
| 4 | 60 分钟覆盖与缺口图 | 120 窗口、缺口、A/B/C/D | 不说全程无丢包 |
| 5 | CSV 筛选 window_index=88 | coverage 0、D、暂不判定 | 不把拒识说成分期错误 |
| 6 | 输出目录 | HTML/MD/JSON/CSV/PNG/manifest | 不展示 synthetic 指标为成绩 |
| 7 | 结论边界 | 公开数据与真实设备数据职责分离 | 不声称 PSG/专家验证 |
"""
    narration = f"""眠卫 MetaSleep-Guard 围绕 MetaBCI 的 Brainda、BrainFlow 和 Brainstim 三部分开发。当前真实实验使用 OpenBCI Cyton 采集 Fp1、Fp2 前额双导。十分钟质量标定数据覆盖率为 {ten['coverage_ratio']:.3%}，系统对睁眼、闭眼及多类伪迹进行质量分级。六十分钟连续实验生成一百二十个三十秒窗口，覆盖率为 {sixty['coverage_ratio']:.3%}，并真实记录到数据中断。第八十八个窗口覆盖率为零，系统将其判为 D 级并输出暂不判定，说明质量守护不会对无效数据强行给出可信结论。每次分析自动导出 HTML、Markdown、JSON、CSV、图表和来源清单。需要强调，OpenBCI 数据用于验证采集、质量审计、中断检测、拒识与报告，不用于证明睡眠分期准确率；当前分期指标来自带专家标签的 Sleep-EDF 数据，ISRUC 目前仅完成数据接口与结构测试。"""
    checklist = r"""# 录屏检查清单

- [ ] 使用 `metabci` 环境运行 status，画面显示三组件 True。
- [ ] 浏览器缩放适中，报告标题、真实数据说明和覆盖率清晰可见。
- [ ] 展示 10 分钟波形、频谱、阶段质量和异常标注。
- [ ] 展示 60 分钟覆盖时间线、缺口、质量趋势和拒识分布。
- [ ] CSV 筛选第 88 窗口，拍到 `coverage_ratio=0`、`quality_grade=D`、`暂不判定`。
- [ ] 展示自动报告目录中的 HTML、MD、JSON、CSV、PNG、manifest。
- [ ] 展示 public_sleep_metrics/README.md 的数据边界。
- [ ] 旁白不出现“确认入睡、PSG 级、全程无丢包、Alpha 成功、OpenBCI 五分类准确率”。
- [ ] 总时长不超过 3 分钟，最后预留 3 秒项目名称画面。
"""
    return [
        _write_text(video_dir / "video_script.md", script),
        _write_text(video_dir / "storyboard.md", storyboard),
        _write_text(video_dir / "narration.txt", narration),
        _write_text(video_dir / "screen_capture_checklist.md", checklist),
    ]


def _export_quality_tables(metric_dir: Path, ten: dict, sixty: dict, ten_rows: list[dict], windows: list[dict]) -> list[Path]:
    outputs: list[Path] = []
    quality_rows = []
    for experiment, summary in (("10-minute calibration", ten), ("60-minute continuous", sixty)):
        for flag, count in summary["artifact_detection_counts"].items():
            quality_rows.append({"experiment": experiment, "quality_flag": flag, "count": count})
    outputs.append(_write_csv(metric_dir / "quality_audit_summary.csv", ["experiment", "quality_flag", "count"], quality_rows))

    stage_rows = []
    for row in ten_rows:
        stage_rows.append(
            {
                "experiment": "10-minute calibration",
                "stage": row["stage_name"],
                "window_count": 1,
                "coverage_ratio": row["coverage_ratio"],
                "sample_completeness_ratio": row["sample_completeness_ratio"],
                "grade_counts": row["quality_grade"],
                "rejected_count": int(row["usable_for_window_inference"] == "False"),
            }
        )
    for row in sixty["stage_quality_statistics"]:
        stage_rows.append(
            {
                "experiment": "60-minute continuous",
                "stage": row["stage_name"],
                "window_count": row["window_count"],
                "coverage_ratio": row["mean_coverage_ratio"],
                "sample_completeness_ratio": row["mean_sample_completeness_ratio"],
                "grade_counts": json.dumps(row["grade_counts"], ensure_ascii=False),
                "rejected_count": row["rejected_window_count"],
            }
        )
    outputs.append(
        _write_csv(
            metric_dir / "stage_quality_statistics.csv",
            ["experiment", "stage", "window_count", "coverage_ratio", "sample_completeness_ratio", "grade_counts", "rejected_count"],
            stage_rows,
        )
    )

    rejected = [row for row in windows if row["usable_for_window_inference"] == "False"]
    outputs.append(_write_csv(metric_dir / "low_quality_windows.csv", list(windows[0]), rejected))
    patterns = ["data_dropout", "effective_data_ratio_insufficient", "zero_variance_channel", "abnormal_amplitude", "motion_artifact", "strong_line_noise"]
    examples = [next(row for row in windows if int(row["window_index"]) == 88)]
    for pattern in patterns:
        match = next((row for row in rejected if pattern in row["quality_flags"] and row not in examples), None)
        if match:
            examples.append(match)
    outputs.append(_write_csv(metric_dir / "typical_bad_quality_cases.csv", list(windows[0]), examples))
    outputs.append(_write_csv(metric_dir / "window_88_evidence.csv", list(windows[0]), [examples[0]]))
    return outputs


def _manifest_rows(
    repo_root: Path,
    reports_root: Path,
    metric_outputs: list[Path],
    public_outputs: list[Path],
    ten: dict,
    sixty: dict,
) -> list[dict]:
    rows = []

    def add(path: Path | str, data_type: str, source: str, purpose: str, provenance: str, accuracy: bool, quality: bool, note: str) -> None:
        rows.append(
            {
                "file_path": str(path),
                "data_type": data_type,
                "source": source,
                "purpose": purpose,
                "real_or_synthetic": provenance,
                "used_for_accuracy": "yes" if accuracy else "no",
                "used_for_quality": "yes" if quality else "no",
                "note": note,
            }
        )

    add(ten["formal_data_file"], "OpenBCI GUI TXT", "SX 10-minute experiment", "quality calibration", "real", False, True, "Ch2/Fp1 and Ch7/Fp2 only")
    for path in sixty["segment_files"]:
        add(path, "OpenBCI GUI TXT segment", "SX 60-minute experiment", "continuous-window integrity", "real", False, True, "part of seven-segment recording")
    for directory, label in ((reports_root / "ten_min_quality_calibration", "10-minute"), (reports_root / "sixty_min_continuous_recording", "60-minute")):
        for name in ("report.html", "report.md", "summary.json"):
            add(directory / name, "generated report", label, "submission evidence", "exported_from_real", False, True, "formal real-data output")
        for path in sorted(directory.glob("*.csv")):
            add(path, "generated CSV", label, "quality/window evidence", "exported_from_real", False, True, "formal real-data output")
        for path in sorted((directory / "figures").glob("*.png")):
            add(path, "generated PNG", label, "visual evidence", "exported_from_real", False, True, "formal real-data output")
    for path in metric_outputs:
        add(path, "submission CSV", "verified report outputs", "metric evidence", "exported_from_real", False, True, "exported by metrics-export")
    for path in public_outputs:
        add(
            path,
            "public_sleep_real_baseline",
            "Sleep-EDF / Sleep Physionet",
            "sleep-staging accuracy evidence",
            "real_public_dataset",
            True,
            False,
            "subject-level split, no epoch leakage",
        )
    for path in sorted((repo_root / "outputs/metasleepguard_outputs/metrics").glob("*.json")):
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("metadata", {}).get("synthetic_demo"):
            add(path, "metric JSON", "synthetic public-sleep smoke", "code-path smoke only", "synthetic", False, False, "smoke test only, not submitted as accuracy evidence")
    loader_note = (
        "real Sleep-EDF baseline present; ISRUC real-data evaluation pending"
        if public_outputs
        else "real public datasets not present"
    )
    add(
        repo_root / "datasets/public_sleep/loaders.py",
        "source code",
        "project",
        "Sleep-EDF/ISRUC interfaces",
        "code_interface",
        False,
        False,
        loader_note,
    )
    return rows


def _copy_public_baseline(repo_root: Path, metric_dir: Path) -> tuple[dict | None, list[Path]]:
    source = repo_root / "outputs/metasleepguard_outputs/public_sleep_real_baseline"
    summary_path = source / "summary.json"
    if not summary_path.exists():
        return None, []
    summary = _read_json(summary_path)
    if not summary.get("real_public_dataset") or summary.get("synthetic_demo"):
        raise RuntimeError("public_sleep_real_baseline summary must identify real non-synthetic data")
    destination = metric_dir / "public_sleep_metrics"
    destination.mkdir(parents=True, exist_ok=True)
    names = [
        "summary.json",
        "sleep_edf_3class_metrics.json",
        "sleep_edf_5class_metrics.json",
        "sleep_edf_subject_split.csv",
        "README.md",
        "DATA_LIMITATIONS.md",
    ]
    copied = []
    for name in names:
        source_path = source / name
        if not source_path.exists():
            raise FileNotFoundError(f"real public baseline output is incomplete: {source_path}")
        destination_path = destination / name
        shutil.copy2(source_path, destination_path)
        copied.append(destination_path)
    for name in ("confusion_matrix_3class.png", "confusion_matrix_5class.png"):
        source_path = source / "figures" / name
        destination_path = destination / name
        shutil.copy2(source_path, destination_path)
        copied.append(destination_path)
    return summary, copied


def generate_submission_materials(
    repo_root: str | Path,
    section: str = "all",
    reports_root: str | Path | None = None,
    submission_root: str | Path | None = None,
) -> dict[str, str]:
    repo_root = Path(repo_root).resolve()
    reports_root = Path(reports_root).resolve() if reports_root else repo_root / "outputs/metasleepguard_outputs/real_openbci_reports"
    submission_root = Path(submission_root).resolve() if submission_root else repo_root / "outputs/metasleepguard_submission"
    ten_dir = reports_root / "ten_min_quality_calibration"
    sixty_dir = reports_root / "sixty_min_continuous_recording"
    ten = _read_json(ten_dir / "summary.json")
    sixty = _read_json(sixty_dir / "summary.json")
    test_result_path = repo_root / "outputs/metasleepguard_outputs/test_results.json"
    test_result = _read_json(test_result_path) if test_result_path.exists() else {}
    windows = _read_csv(sixty_dir / "thirty_sec_window_integrity.csv")
    ten_rows = _read_csv(ten_dir / "window_or_stage_quality.csv")
    window_88 = _validate_window_88(windows)
    outputs: list[Path] = []

    if section in {"all", "metrics"}:
        metric_dir = submission_root / "03_metric_validation_data"
        public_summary, public_outputs = _copy_public_baseline(repo_root, metric_dir)
        outputs.append(_write_text(repo_root / "outputs/metasleepguard_outputs/current_runnable_tasks.md", _runnable_tasks_markdown(public_summary)))
        outputs.append(_write_text(repo_root / "outputs/metasleepguard_outputs/real_data_conclusion_for_submission.md", _real_conclusion(ten, sixty, window_88)))
        outputs.append(_write_text(repo_root / "outputs/metasleepguard_outputs/public_sleep_metrics/README.md", _public_sleep_readme(public_summary)))
        outputs.append(_write_text(metric_dir / "REAL_DATA_RUNNABLE_ITEMS.md", _real_items(ten, sixty, public_summary)))
        metric_outputs = _export_quality_tables(metric_dir, ten, sixty, ten_rows, windows)
        outputs.extend(metric_outputs)
        outputs.extend(public_outputs)
        manifest = _write_csv(
            metric_dir / "manifest.csv",
            ["file_path", "data_type", "source", "purpose", "real_or_synthetic", "used_for_accuracy", "used_for_quality", "note"],
            _manifest_rows(repo_root, reports_root, metric_outputs, public_outputs, ten, sixty),
        )
        outputs.append(manifest)

    if section == "all":
        outputs.append(
            _write_text(
                submission_root / "04_objective_score_evidence/objective_score_evidence.md",
                _objective_evidence(repo_root, reports_root, public_summary),
            )
        )
        outputs.append(
            _write_text(
                submission_root / "01_project_test_document/test_document_content.md",
                _test_document(ten, sixty, test_result, public_summary),
            )
        )

    if section in {"all", "demo"}:
        outputs.extend(_video_assets(submission_root / "02_demo_video_assets", ten, sixty))

    return {path.name: str(path) for path in outputs}


__all__ = ["generate_submission_materials"]
