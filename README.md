# MetaSleep-Guard

MetaSleep-Guard is a two-channel trusted sleep-staging and signal-quality auditing system built around the MetaBCI ecosystem. The project integrates Brainda-compatible public-data modeling, BrainFlow/OpenBCI acquisition, and Brainstim calibration tasks.

The implementation includes Sleep-EDF public-data adapters, leakage-safe subject-level splits, dual-channel EEG feature extraction, RandomForest small-sample baselines, signal-quality auditing, trusted abstention, OpenBCI file replay, visualization, and HTML report generation.

## Repository

This repository contains the source code, environment configuration, test scripts, report-generation modules, quality-audit modules, and evaluation workflows used for the MetaBCI Innovation Application Competition preliminary submission.

## Environment

Recommended environment:

- Windows 11
- Python / conda environment: metabci
- MetaBCI components: Brainda, BrainFlow, Brainstim
- OpenBCI Cyton data input and OpenBCI GUI TXT file replay

## Smoke Test

Run the following command in the project root directory:

    python -m MetaSleepGuard.tests.run_smoke_tests

Or use the unified PowerShell launcher:

    .\run.ps1 -Task test
    .\run.ps1 -Task status
    .\run.ps1 -Task real-openbci-report
    .\run.ps1 -Task public-sleep-real-baseline

## Main Commands

    .\run.ps1 -Task status
    .\run.ps1 -Task test
    .\run.ps1 -Task real-openbci-report
    .\run.ps1 -Task openbci-file-replay
    .\run.ps1 -Task public-sleep-real-baseline
    .\run.ps1 -Task metrics-export
    .\run.ps1 -Task submission-pack
    .\run.ps1 -Task demo-assets

## Evaluation Note

The submitted public Sleep-EDF metrics are based on a 5-subject, one-night-per-subject RandomForest small-sample baseline with GroupKFold subject-level split.

The real OpenBCI data are used for acquisition-chain validation, file replay, 30-second window integrity analysis, signal-quality auditing, trusted abstention, and automatic report validation. The OpenBCI data are not used as sleep-staging accuracy evidence.

## License

This project is released under the GNU General Public License v2.0. Third-party datasets, packages, and tools retain their respective licenses and usage terms.
