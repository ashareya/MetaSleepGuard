# MetaSleep-Guard Dual-Environment Guide

## Environment A: metabci

Python：

    <conda-root>\envs\metabci\python.exe

Main functions：

- MetaBCI BrainFlow
- MetaBCI Brainda
- MNE and Sleep-EDF processing
- model evaluation
- OpenBCI replay
- quality auditing
- trusted rejection
- reports
- tests

## Environment B: metabci_stim

Python：

    <conda-root>\envs\metabci_stim\python.exe

Main functions：

- MetaBCI Brainstim
- PsychoPy
- visual stimulation
- calibration
- experiment prompts
- marker logging

## Commands

    $py = "<conda-root>\envs\metabci\python.exe"
    $stimPy = "<conda-root>\envs\metabci_stim\python.exe"

    .\run.ps1 -Task status -Python $py
    .\run.ps1 -Task test -Python $py
    .\run.ps1 -Task metabci-integration-test -Python $py
    .\run.ps1 -Task brainstim -Synthetic -Python $stimPy

`run.ps1` also detects these two environments automatically when `-Python` is omitted. An explicitly selected Python is rejected early if it cannot import the required MetaBCI module.

## Verified Results

- Analysis environment imports `metabci`, `metabci.brainflow`, and `metabci.brainda`.
- Brainstim environment imports `psychopy` and `metabci.brainstim`.
- MetaBCI integration test passes.
- Brainstim dry-run passes.
- Automated test result: 78 passed, 0 failed.

## Design Rationale

PsychoPy and its graphical dependencies are isolated from the analysis environment. This preserves the stability of MNE, Sleep-EDF processing, model evaluation, and automated tests while retaining full Brainstim functionality in the dedicated stimulus environment.
