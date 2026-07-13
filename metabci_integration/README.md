# MetaBCI Core Framework Integration Layer

This directory makes the MetaBCI framework boundary explicit for MetaSleep-Guard.

The local probe must use the actual project Python environment, for example:

```powershell
$py = "C:\Users\ZYH\anaconda3\envs\metabci\python.exe"
.\run.ps1 -Task status -Python $py
.\run.ps1 -Task metabci-integration-test -Python $py
```

## Components

- `metabci_component_check.py` imports `metabci` and discovers the locally importable `metabci.*` module tree. It reports `metabci.brainflow`, `metabci.brainda`, and `metabci.brainstim` using the real import result from the current Python environment.
- `metabci_brainflow_adapter.py` aligns OpenBCI file replay and BrainFlow/Cyton acquisition with MetaBCI BrainFlow. The smoke test imports `metabci.brainflow.*` and exercises `metabci.brainflow.amplifiers.RingBuffer`.
- `metabci_brainstim_adapter.py` aligns calibration prompts, event markers, LSL marker output, and CSV logs with the MetaBCI Brainstim platform boundary. If `metabci.brainstim` is not importable because optional PsychoPy dependencies are missing, the adapter records that exact error and still validates marker logging.
- `metabci_brainda_adapter.py` aligns public Sleep-EDF/ISRUC processing, feature extraction, subject-level splits, and evaluation with MetaBCI Brainda. The smoke test imports Brainda modules and exercises `EnhancedLeaveOneGroupOut` for subject-level split checks.

## Local Probe Result

On the specified local environment, `metabci`, `metabci.brainflow`, and `metabci.brainda` import successfully. `metabci.brainstim` package files are present, but the analysis environment blocks import because `psychopy` is not installed. This is recorded as an unavailable optional Brainstim runtime, not hidden or faked.

## Project Additions Above MetaBCI Basics

MetaSleep-Guard keeps the original real OpenBCI reports, Sleep-EDF small-sample baseline, quality audit, trusted abstention, and automatic reporting. The application-specific additions are:

- Sleep quality audit rules for 30-second windows.
- 30-second window integrity and file-gap checks.
- Trusted rejection with "do not decide" output on low-quality or low-confidence data.
- Automated HTML/Markdown/CSV/JSON reports for real OpenBCI and public Sleep-EDF runs.
