# MetaBCI Core Usage Explanation

MetaSleep-Guard contains a dedicated `metabci_integration/` layer. The integration layer probes and calls APIs that are present in the actual local MetaBCI installation; it does not create nonexistent MetaBCI interfaces.

## Dual-Environment Runtime

### Analysis environment

Python:

    <conda-root>\envs\metabci\python.exe

Verified components:

- `metabci`
- `metabci.brainflow`
- `metabci.brainda`

Responsibilities:

- OpenBCI acquisition-chain alignment
- OpenBCI file replay
- Sleep-EDF processing
- subject-level model evaluation
- quality auditing
- trusted abstention
- report generation
- automated tests

### Brainstim environment

Python:

    <conda-root>\envs\metabci_stim\python.exe

Verified components:

- `psychopy`
- `metabci.brainstim`
- Brainstim dry-run calibration

Responsibilities:

- visual stimulus presentation
- calibration paradigms
- experiment prompts
- countdown
- LSL markers
- event logs

The separation is intentional. PsychoPy remains in the stimulus environment to avoid destabilizing the analysis environment and its MNE, public sleep-data, and evaluation dependencies.

## BrainFlow Usage

The BrainFlow adapter imports MetaBCI BrainFlow modules and exercises `metabci.brainflow.amplifiers.RingBuffer`. It connects the MetaBCI framework boundary to OpenBCI Cyton acquisition and OpenBCI GUI file replay.

## Brainda Usage

The Brainda adapter imports MetaBCI Brainda modules and calls `EnhancedLeaveOneGroupOut` to verify subject-level separation between training and testing data.

## Brainstim Usage

The Brainstim adapter is verified in the `metabci_stim` environment. The environment imports `metabci.brainstim` and PsychoPy, and the project Brainstim dry-run calibration completes successfully.

## Project Extensions

On top of MetaBCI base functions, the project adds:

- sleep signal-quality auditing
- 30-second window-integrity checks
- trusted rejection for unusable windows
- real OpenBCI quality reports
- Sleep-EDF baseline reports
- HTML, Markdown, CSV, JSON, and figure exports

## Verification

Analysis environment:

    $py = "<conda-root>\envs\metabci\python.exe"

    .\run.ps1 -Task status -Python $py
    .\run.ps1 -Task test -Python $py
    .\run.ps1 -Task metabci-integration-test -Python $py

Brainstim environment:

    $stimPy = "<conda-root>\envs\metabci_stim\python.exe"

    .\run.ps1 -Task brainstim -Synthetic -Python $stimPy
