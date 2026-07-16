# MetaSleep-Guard

MetaSleep-Guard is a two-channel trusted sleep-staging and signal-quality auditing system built on the MetaBCI core framework.

本项目基于 MetaBCI 的 BrainFlow、Brainstim 和 Brainda 子平台能力构建，并通过独立集成层将 MetaBCI 基础功能与双导睡眠监测应用连接起来。

## MetaBCI Core Usage

- **MetaBCI / BrainFlow**：用于 OpenBCI Cyton 数据接入、OpenBCI GUI 文件回放、在线缓存和 30 秒窗口运行链路。集成测试实际导入 `metabci.brainflow`，并调用 `metabci.brainflow.amplifiers.RingBuffer`。
- **MetaBCI / Brainda**：用于公开睡眠数据处理流程、被试级划分检查和模型评价组织。集成测试实际导入 `metabci.brainda`，并调用 `EnhancedLeaveOneGroupOut` 检查训练集和测试集被试无重叠。
- **MetaBCI / Brainstim**：用于刺激提示、倒计时、标定范式、事件标记和 LSL marker。Brainstim 图形功能在独立的 `metabci_stim` 环境中运行。
- **项目新增功能**：睡眠信号质量审计、30 秒窗口完整性统计、可信拒识、真实 OpenBCI 报告和自动化指标报告。

## Dual-Environment Design

本项目采用两个相互隔离的 conda 环境，不将 PsychoPy 强行安装到分析环境中。

### 1. Analysis environment

用途：

- MetaBCI BrainFlow 和 Brainda 集成
- Sleep-EDF 数据处理
- 模型训练与评估
- OpenBCI 文件回放
- 信号质量审计
- 可信拒识
- 报告生成
- 单元测试和集成测试

Python：

    C:\Users\ZYH\anaconda3\envs\metabci\python.exe

已验证：

- `metabci` 可导入
- `metabci.brainflow` 可导入
- `metabci.brainda` 可导入
- 69 项测试通过

### 2. Brainstim environment

用途：

- MetaBCI Brainstim
- PsychoPy 图形刺激
- 标定范式
- 实验提示与倒计时
- 事件标记和 marker 日志

Python：

    C:\Users\ZYH\anaconda3\envs\metabci_stim\python.exe

已验证：

- `psychopy` 可导入
- `metabci.brainstim` 可导入
- Brainstim dry-run 标定流程通过

双环境分工是有意的工程设计。PsychoPy 及其图形依赖较重，单独使用刺激环境可以避免影响 MNE、Sleep-EDF 数据处理和既有分析测试环境的稳定性。

完整说明见 `DUAL_ENVIRONMENT_GUIDE.md`。

## Verification Commands

在项目根目录运行：

    $py = "C:\Users\ZYH\anaconda3\envs\metabci\python.exe"
    $stimPy = "C:\Users\ZYH\anaconda3\envs\metabci_stim\python.exe"

    .\run.ps1 -Task status -Python $py
    .\run.ps1 -Task test -Python $py
    .\run.ps1 -Task metabci-integration-test -Python $py

    .\run.ps1 -Task brainstim -Synthetic -Python $stimPy

## Main Commands

    .\run.ps1 -Task status -Python $py
    .\run.ps1 -Task test -Python $py
    .\run.ps1 -Task metabci-integration-test -Python $py
    .\run.ps1 -Task real-openbci-report -Python $py
    .\run.ps1 -Task openbci-file-replay -Python $py
    .\run.ps1 -Task public-sleep-download -DataRoot .\data\public_sleep\sleep_edf_raw -MaxSubjects 15 -Python $py
    .\run.ps1 -Task public-sleep-real-baseline -DataRoot .\data\public_sleep\sleep_edf_raw -MaxSubjects 15 -Python $py
    .\run.ps1 -Task metrics-export -Python $py
    .\run.ps1 -Task submission-pack -Python $py
    .\run.ps1 -Task demo-assets -Python $py
    .\run.ps1 -Task brainstim -Synthetic -Python $stimPy

## Evaluation Evidence

The current Sleep-EDF metrics are based on 15 subjects, one night per subject, and subject-level GroupKFold evaluation splits.

- Valid 30-second epochs: 15,029
- Three-class Accuracy: 0.853483
- Three-class Macro-F1: 0.806407
- Five-class Accuracy: 0.801650
- Five-class Macro-F1: 0.713587
- Real OpenBCI 10-minute coverage: 0.993948
- Real OpenBCI 60-minute coverage: 0.973399

Public expert-labeled Sleep-EDF data are used for sleep-staging performance evaluation. Real OpenBCI data are used for acquisition-chain validation, file replay, window-integrity analysis, quality auditing, trusted abstention, and automatic report validation. OpenBCI data are not used as sleep-staging accuracy evidence.

## License

This project is released under the GNU General Public License v2.0. Third-party datasets, packages, MetaBCI components, and tools retain their respective licenses and usage terms.
