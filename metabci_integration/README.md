# MetaBCI Core Framework Integration Layer

This directory contains the explicit MetaBCI integration boundary for MetaSleep-Guard.

## Components

- `metabci_component_check.py`:discovers the locally available MetaBCI module tree.
- `metabci_brainflow_adapter.py`:imports MetaBCI BrainFlow and exercises RingBuffer.
- `metabci_brainda_adapter.py`:imports MetaBCI Brainda and verifies subject-level split behavior.
- `metabci_brainstim_adapter.py`:checks Brainstim availability and connects the project calibration workflow to the MetaBCI Brainstim runtime boundary.

## Two Runtime Environments

### Analysis

    <conda-root>\envs\metabci\python.exe

Used for BrainFlow, Brainda, Sleep-EDF analysis, quality auditing, reports, and automated testing.

### Brainstim

    <conda-root>\envs\metabci_stim\python.exe

Used for PsychoPy, MetaBCI Brainstim, visual stimulation, calibration, and event markers.

## Verification Commands

    $py = "<conda-root>\envs\metabci\python.exe"
    $stimPy = "<conda-root>\envs\metabci_stim\python.exe"

    .\run.ps1 -Task status -Python $py
    .\run.ps1 -Task metabci-integration-test -Python $py
    .\run.ps1 -Task brainstim -Synthetic -Python $stimPy

The analysis environment and Brainstim environment are intentionally separated to preserve dependency stability.
