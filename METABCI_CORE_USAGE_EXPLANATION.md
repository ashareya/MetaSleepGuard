# MetaBCI Core Usage Explanation

原版本主要调用底层依赖库；修订后已补充 MetaBCI 核心框架集成层，并在代码与测试命令中体现 MetaBCI 基础功能调用。

## What Changed

MetaSleep-Guard now has a dedicated `MetaSleepGuard/metabci_integration/` layer. This layer does not invent or fake MetaBCI APIs. It probes the actual local Python environment and only reports modules that are really importable.

Local probe result in `C:\Users\ZYH\anaconda3\envs\metabci\python.exe`:

- `metabci`: importable.
- `metabci.brainflow`: importable.
- `metabci.brainda`: importable.
- `metabci.brainstim`: package files are present, but import currently fails because `psychopy` is not installed in this analysis environment.

## BrainFlow Usage

The project uses MetaBCI/BrainFlow to align OpenBCI acquisition and file replay:

- Real-time OpenBCI/Cyton and file replay remain in `realtime/`.
- The new adapter imports `metabci.brainflow`, `metabci.brainflow.amplifiers`, `metabci.brainflow.workers`, and `metabci.brainflow.logger`.
- The integration test exercises `metabci.brainflow.amplifiers.RingBuffer` as a lightweight MetaBCI BrainFlow base function.

## Brainstim Usage

The project uses the MetaBCI/Brainstim platform boundary for calibration paradigms, prompts, event markers, and LSL marker logging:

- The adapter probes `metabci.brainstim` and records the current `psychopy` import blocker.
- The project Brainstim task still runs a dry-run marker smoke test through `MetaSleepGuard.brainstim_task.calibration_task.run_calibration_task`.
- If the runtime is switched to an environment with PsychoPy installed, the same adapter can report Brainstim as importable.

## Brainda Usage

The project uses MetaBCI/Brainda concepts and available interfaces for public sleep-data processing and evaluation:

- Sleep-EDF/ISRUC loaders remain project-specific because this local MetaBCI install does not expose dedicated sleep-staging datasets.
- The adapter imports `metabci.brainda.datasets`, `metabci.brainda.paradigms`, `metabci.brainda.algorithms.feature_analysis.freq_analysis`, and `metabci.brainda.algorithms.utils.model_selection`.
- The integration test calls `EnhancedLeaveOneGroupOut` to verify subject-level no-overlap split behavior.

## Project Additions

On top of MetaBCI base functions, MetaSleep-Guard adds:

- Sleep quality audit.
- 30-second window integrity checks.
- Trusted rejection / abstention for low-quality or low-confidence windows.
- Automatic HTML, Markdown, CSV, JSON, and figure reports.
- Real OpenBCI report generation and Sleep-EDF small-sample baseline evidence.

## Required Verification Command

```powershell
$py = "C:\Users\ZYH\anaconda3\envs\metabci\python.exe"
.\run.ps1 -Task metabci-integration-test -Python $py
```

Expected output includes MetaBCI core import success, BrainFlow adapter success, Brainda adapter success, Brainstim import availability, and Brainstim marker-log smoke success.
