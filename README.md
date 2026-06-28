# MetaSleep-Guard

MetaSleep-Guard is a two-channel trusted sleep-staging system built around the
MetaBCI ecosystem: Brainda-compatible public-data modeling, BrainFlow/OpenBCI
acquisition, and Brainstim calibration tasks.

The implementation includes Sleep-EDF and ISRUC-Sleep adapters, leakage-safe
subject splits, causal context features, XGBoost-compatible baselines, signal
quality rules, calibrated active rejection, file replay, live Cyton streaming,
visualization, and HTML reports.

## Smoke Test

```powershell
python -m MetaSleepGuard.tests.run_smoke_tests
```

Or use the unified PowerShell launcher:

```powershell
.\MetaSleepGuard\run.ps1 -Task test
.\MetaSleepGuard\run.ps1 -Task status
.\run.ps1 -Task real-openbci-report
.\MetaSleepGuard\run.ps1 -Task train -Dataset sleep-edf -ClassificationTask 5class
```

## Main Commands

```powershell
python -m MetaSleepGuard.experiments.run_train --dataset sleep-edf --task 5class
python -m MetaSleepGuard.experiments.run_cross_dataset --task 5class
python -m MetaSleepGuard.experiments.run_bdf_fif_audit --input-dir D:\data\boruikang
python -m MetaSleepGuard.experiments.run_openbci_realtime --synthetic --duration-sec 31
python -m MetaSleepGuard.experiments.run_generate_report
```

Synthetic data is used only when a public dataset path is absent. It validates
the software path, not sleep-stage accuracy. Unlabeled Boruikang/OpenBCI data is
never treated as expert sleep-stage ground truth.

Each audit, replay, real-time, and report run writes to a unique timestamped
directory under `_codex_tmp/metasleepguard_outputs/`. Synthetic runs are marked
as `synthetic_demo=True` in terminal output and metric metadata.

See [docs/README.md](docs/README.md) for setup and complete command examples.
