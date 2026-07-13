# MetaSleep-Guard

MetaSleep-Guard is a two-channel trusted sleep-staging and signal-quality auditing system built on the MetaBCI core framework. The project is now organized around explicit MetaBCI BrainFlow, Brainstim, and Brainda integration layers, not only around direct calls to BrainFlow, sklearn, MNE, or other lower-level dependencies.

本项目基于 MetaBCI 的 BrainFlow、Brainstim、Brainda 子平台能力构建：

- 使用 MetaBCI/BrainFlow 完成 OpenBCI 数据接入、文件回放、在线缓存和 30 秒窗口运行链路对齐。代码实际导入 `metabci.brainflow`，并在集成测试中调用 `metabci.brainflow.amplifiers.RingBuffer`。
- 使用 MetaBCI/Brainstim 完成标定流程、刺激提示、事件标记和 LSL marker 对齐。当前 `metabci` 分析环境中 `metabci.brainstim` 因缺少 `psychopy` 无法直接导入，集成层会记录真实错误，并运行项目 Brainstim 标定 marker 日志 smoke；在安装 PsychoPy 的 stim 环境中可启用图形刺激任务。
- 使用 MetaBCI/Brainda 思路和可用接口完成公开睡眠数据处理、被试级划分和模型评估。代码实际导入 `metabci.brainda`，并在集成测试中调用 `metabci.brainda.algorithms.utils.model_selection.EnhancedLeaveOneGroupOut` 做被试级划分检查。
- 在 MetaBCI 基础能力之上，本项目新增睡眠质量审计、30 秒窗口完整性、可信拒识和自动报告模块。

## Repository

This repository contains the source code, environment configuration, test scripts, report-generation modules, quality-audit modules, MetaBCI integration adapters, and evaluation workflows used for the MetaBCI Innovation Application Competition preliminary submission.

## Environment

Recommended environment:

- Windows 11
- Python / conda environment: `metabci`
- Python executable used for verification: `C:\Users\ZYH\anaconda3\envs\metabci\python.exe`
- MetaBCI components checked at runtime: `metabci`, `metabci.brainflow`, `metabci.brainda`, `metabci.brainstim`
- OpenBCI Cyton data input and OpenBCI GUI TXT file replay

## Smoke Test

Run the following command in the project root directory:

```powershell
python -m MetaSleepGuard.tests.run_smoke_tests
```

Or use the unified PowerShell launcher:

```powershell
$py = "C:\Users\ZYH\anaconda3\envs\metabci\python.exe"
.\run.ps1 -Task status -Python $py
.\run.ps1 -Task test -Python $py
.\run.ps1 -Task metabci-integration-test -Python $py
.\run.ps1 -Task real-openbci-report -Python $py
.\run.ps1 -Task public-sleep-real-baseline -Python $py
```

`.\run.ps1 -Task metabci-integration-test -Python $py` will import MetaBCI core and available submodules, exercise MetaBCI BrainFlow and Brainda base functionality, and print Brainstim availability plus marker-log smoke results.

## Main Commands

```powershell
.\run.ps1 -Task status -Python $py
.\run.ps1 -Task test -Python $py
.\run.ps1 -Task metabci-integration-test -Python $py
.\run.ps1 -Task real-openbci-report -Python $py
.\run.ps1 -Task openbci-file-replay -Python $py
.\run.ps1 -Task public-sleep-real-baseline -Python $py
.\run.ps1 -Task metrics-export -Python $py
.\run.ps1 -Task submission-pack -Python $py
.\run.ps1 -Task demo-assets -Python $py
```

## Evaluation Note

The submitted public Sleep-EDF metrics are based on a 5-subject, one-night-per-subject RandomForest small-sample baseline with subject-level splits.

The real OpenBCI data are used for MetaBCI/BrainFlow acquisition-chain validation, file replay, 30-second window integrity analysis, signal-quality auditing, trusted abstention, and automatic report validation. The OpenBCI data are not used as sleep-staging accuracy evidence.

## License

This project is released under the GNU General Public License v2.0. Third-party datasets, packages, MetaBCI components, and tools retain their respective licenses and usage terms.
