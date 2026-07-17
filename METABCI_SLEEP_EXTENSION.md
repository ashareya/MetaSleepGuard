# metabci_sleep：MetaBCI兼容睡眠扩展包

`metabci_sleep` 0.2.0 是由 MetaSleep-Guard 维护的可安装扩展包。它遵循本机
MetaBCI 的真实 Dataset、Paradigm 和 ProcessWorker 接口，并复用项目已有的
特征、模型、质量审计、可信拒识、实时处理和报告实现。

它没有修改 MetaBCI 官方源码，也尚未合并进 MetaBCI 官方仓库。因此准确称谓是
“MetaBCI兼容扩展包”，不是“MetaBCI官方新增功能”。

## 安装与验证

在 MetaSleepGuard 仓库根目录运行：

```powershell
$py = "C:\Users\ZYH\anaconda3\envs\metabci\python.exe"
& $py -m pip install .
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 -Task metabci-sleep-smoke -Python $py
```

公开接口：

```python
from metabci_sleep.datasets import SleepEDF, ISRUCSleep
from metabci_sleep.paradigms import SleepStaging
from metabci_sleep.algorithms import (
    SleepFeatureExtractor, SleepStagingEstimator,
    ProbabilityCalibrator, CoverageRiskEvaluator,
    SleepQualityAuditor, TrustedRejector, SleepMetrics,
)
from metabci_sleep.realtime import WindowIntegrityAuditor, OpenBCISleepWorker
from metabci_sleep.reporting import SleepReportBuilder
from metabci_sleep.brainstim import SleepCalibrationProtocol
```

## 兼容关系

| 扩展组件 | MetaBCI接口或项目实现 | 作用 |
|---|---|---|
| `SleepEDF` | 继承 `metabci.brainda.datasets.base.BaseDataset` | 返回标准subject/session/run/MNE Raw结构 |
| `SleepStaging` | 继承 `metabci.brainda.paradigms.base.BaseParadigm` | 生成30秒三/四/五分类睡眠窗口 |
| `SleepQualityAuditor` | 包装 `MetaSleepGuard.quality.quality_audit` | 输出质量分数、等级、伪迹和可靠性 |
| `TrustedRejector` | 包装 `MetaSleepGuard.rejection.ActiveRejector` | 低质量或低置信度时输出“暂不判定” |
| `SleepFeatureExtractor` | 组合现有特征模块 | 多维特征和仅使用历史窗口的因果上下文 |
| `SleepStagingEstimator` | 组合XGBoost/随机森林回退 | 训练、预测、概率输出和模型保存 |
| `OpenBCISleepWorker` | 继承MetaBCI `ProcessWorker` | MetaBCI在线生命周期中的30秒处理 |
| `SleepMetrics` | 新增透明工程规则 | 睡眠效率、潜伏期、WASO、阶段比例和工程评分 |
| `SleepReportBuilder` | 包装现有报告模块 | HTML、Markdown、CSV和JSON统一输出 |

`SleepEDF` 会把 Sleep-EDF 的长区间专家标注展开为30秒数字事件，以兼容 MetaBCI
基于 MNE annotations 的数据结构。`SleepStaging.get_data()` 返回：

- `X`: `(epochs, channels, samples)`，单位为微伏；
- `y`: 编码后的三/四/五分类标签；
- `meta`: subject、session、run、stage、epoch_index和dataset等字段。

## 与官方仓库的边界

当前扩展位于 MetaSleep-Guard 仓库，不能表述为已经加入 MetaBCI 官方功能。
未来迁移到 MetaBCI Fork 时建议映射为：

- `SleepEDF` → `metabci.brainda.datasets`
- `SleepStaging` → `metabci.brainda.paradigms`
- `SleepQualityAuditor` → `metabci.brainda.algorithms.quality`
- `TrustedRejector` → `metabci.brainda.algorithms.rejection`
- `OpenBCISleepWorker` → `metabci.brainflow.workers`

## 生医赛能力迁移矩阵

| 生医赛能力 | 当前状态 |
|---|---|
| 多维时频、Hjorth、熵、分形特征 | 已有代码并进入扩展包 |
| 前两个Epoch因果上下文 | 已有代码并进入扩展包 |
| XGBoost/随机森林、概率校准、覆盖率风险 | 已有代码并进入扩展包 |
| Sleep-EDF、ISRUC和三/四/五分类 | 本轮完成MetaBCI兼容封装 |
| 实时Worker、窗口完整性、报告与Brainstim标定 | 本轮完成MetaBCI兼容封装 |
| 睡眠效率、潜伏期、WASO、阶段比例 | 本轮新增可测试实现 |
| 0–100睡眠评分 | 本轮新增工程启发式；不是临床量表 |

`SleepStagingEstimator.predict_proba()`用于独立批次预测；
`predict_proba_stream()`会跨调用保留同一受试者的因果历史，
切换记录或重新开始实验时调用`reset_stream()`。

ISRUC接口已经通过官方`.rec`结构、双评分标签、标准结构和范式测试；当前正式准确率证据来自30名
Sleep-EDF被试。没有真实ISRUC文件时，不使用模拟数据声称跨数据集性能。

只有提交 Pull Request 并被维护者合并后，才能称为 MetaBCI 官方新增功能。
