# CHANGELOG: MetaBCI Core Framework Integration

## Summary

原版本主要调用底层依赖库;修订后已补充 MetaBCI 核心框架集成层,并在代码与测试命令中体现 MetaBCI 基础功能调用。

## Changes

- Replaced the old single-file `metabci_integration.py` with the package `metabci_integration/`.
- Added `metabci_component_check.py` to import `metabci`, discover the local `metabci.*` module tree, and report BrainFlow, Brainstim, and Brainda availability using real local import results.
- Added `metabci_brainflow_adapter.py` to align OpenBCI/BrainFlow acquisition and replay with MetaBCI BrainFlow. The integration smoke calls `metabci.brainflow.amplifiers.RingBuffer`.
- Added `metabci_brainstim_adapter.py` to align calibration prompts, events, LSL markers, and CSV logs with MetaBCI Brainstim. In this analysis environment, `metabci.brainstim` package files are present but import is blocked by missing `psychopy`; the adapter reports the exact error and still validates marker logging.
- Added `metabci_brainda_adapter.py` to align public Sleep-EDF/ISRUC processing, subject-level splitting, and model evaluation with MetaBCI Brainda. The integration smoke calls `metabci.brainda.algorithms.utils.model_selection.EnhancedLeaveOneGroupOut`.
- Added `experiments/run_metabci_integration_test.py` and `run.ps1 -Task metabci-integration-test`.
- Strengthened `run.ps1 -Task status` so it prints MetaBCI core import status, subplatform checks, and a module-tree summary.
- Updated `README.md`, `docs/README.md`, and generated-submission command metadata to include:
  - `.\run.ps1 -Task status -Python $py`
  - `.\run.ps1 -Task test -Python $py`
  - `.\run.ps1 -Task metabci-integration-test -Python $py`

## Preserved Existing Functions

- Real OpenBCI report generation is preserved.
- Sleep-EDF small-sample public baseline is preserved.
- Quality audit, 30-second window integrity, trusted rejection, and automatic report modules are preserved.
- Existing tests remain routed through `.\run.ps1 -Task test -Python $py`.

## Dual-Environment Runtime

- Documented the separate `metabci` analysis environment.
- Documented the separate `metabci_stim` Brainstim/PsychoPy environment.
- Added verified Brainstim import and dry-run commands.
- Added `DUAL_ENVIRONMENT_GUIDE.md`.