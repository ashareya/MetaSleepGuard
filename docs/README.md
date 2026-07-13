# MetaSleep-Guard Documentation

MetaSleep-Guard is a two-channel EEG sleep-monitoring application built on MetaBCI BrainFlow, Brainstim, and Brainda capabilities.

## Runtime Architecture

项目采用双环境分工。

### Analysis environment: metabci

Python：

    C:\Users\ZYH\anaconda3\envs\metabci\python.exe

负责：

- `metabci.brainflow`
- `metabci.brainda`
- Sleep-EDF 加载和评估
- OpenBCI 文件回放
- 质量审计
- 可信拒识
- 报告生成
- 全量自动测试

运行：

    $py = "C:\Users\ZYH\anaconda3\envs\metabci\python.exe"

    .\run.ps1 -Task status -Python $py
    .\run.ps1 -Task test -Python $py
    .\run.ps1 -Task metabci-integration-test -Python $py
    .\run.ps1 -Task public-sleep-real-baseline -Python $py
    .\run.ps1 -Task real-openbci-report -Python $py

### Stimulus environment: metabci_stim

Python：

    C:\Users\ZYH\anaconda3\envs\metabci_stim\python.exe

负责：

- `metabci.brainstim`
- PsychoPy 图形刺激
- 标定范式
- 提示与倒计时
- LSL marker
- 事件日志

运行：

    $stimPy = "C:\Users\ZYH\anaconda3\envs\metabci_stim\python.exe"

    .\run.ps1 -Task brainstim -Synthetic -Python $stimPy

该环境已实测可导入 `psychopy` 和 `metabci.brainstim`，Brainstim dry-run 已通过。

## Why Two Environments

PsychoPy 包含较重的图形和多媒体依赖。将其与分析环境隔离，可以降低对 MNE、Sleep-EDF 数据处理、模型评价和自动测试环境的影响。分析环境中 Brainstim 显示 unavailable，表示该解释器没有 PsychoPy，并不表示项目的 Brainstim 功能不可用。

## MetaBCI Integration

- BrainFlow adapter：调用 MetaBCI BrainFlow 基础模块及 RingBuffer。
- Brainda adapter：调用 MetaBCI Brainda 模型选择工具完成被试级划分检查。
- Brainstim adapter：在 `metabci_stim` 环境中导入 MetaBCI Brainstim，并运行标定 dry-run。
- Project extensions：质量审计、窗口完整性、可信拒识和自动报告。

## Main Modules

- `metabci_integration/`
- `brainstim_task/`
- `datasets/public_sleep/`
- `preprocessing/`
- `features/`
- `models/`
- `quality/`
- `rejection/`
- `realtime/`
- `reports/`
- `tests/`

## License

Project code is released under GPL-2.0. MetaBCI and all third-party dependencies retain their respective licenses.