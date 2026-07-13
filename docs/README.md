# MetaSleep-Guard Documentation

MetaSleep-Guard is a MetaBCI-based two-channel EEG system for trusted sleep staging and signal-quality guarding. 本项目基于 MetaBCI 的 BrainFlow、Brainstim、Brainda 子平台能力构建，并在仓库中新增了 `MetaSleepGuard/metabci_integration/` 作为核心框架集成层。

## MetaBCI Usage

- MetaBCI/BrainFlow: OpenBCI Cyton acquisition, OpenBCI GUI TXT file replay, runtime chunks, ring-buffer alignment, and 30-second inference windows. The integration test imports `metabci.brainflow.*` and exercises `metabci.brainflow.amplifiers.RingBuffer`.
- MetaBCI/Brainstim: calibration paradigms, stimulus prompts, event markers, LSL marker output, and CSV marker logs. In the current analysis environment, `metabci.brainstim` is detected but import is blocked by missing `psychopy`; this is reported honestly, and the project Brainstim adapter still runs a marker-log smoke test.
- MetaBCI/Brainda: public sleep-data processing, feature/evaluation organization, subject-level split checks, and model-evaluation flow. The integration test imports `metabci.brainda.*` and calls `EnhancedLeaveOneGroupOut` for no-subject-overlap validation.
- Project additions: sleep quality audit, 30-second window integrity, trusted rejection, real OpenBCI report generation, and public Sleep-EDF small-sample baseline reports.

## Quick Start

From the repository root:

```powershell
$py = "C:\Users\ZYH\anaconda3\envs\metabci\python.exe"
.\run.ps1 -Task status -Python $py
.\run.ps1 -Task metabci-integration-test -Python $py
.\run.ps1 -Task test -Python $py
```

The integration-test command must show MetaBCI core import status, BrainFlow/Brainda base-function smoke results, and Brainstim availability/marker-log results:

```powershell
.\run.ps1 -Task metabci-integration-test -Python $py
```

Direct module commands remain available:

```powershell
python -m MetaSleepGuard.experiments.run_train --dataset sleep-edf --task 5class
python -m MetaSleepGuard.experiments.run_eval --dataset sleep-edf --task 5class
python -m MetaSleepGuard.experiments.run_cross_dataset --task 5class
python -m MetaSleepGuard.experiments.run_openbci_file_replay --file D:\data\openbci\record.csv
python -m MetaSleepGuard.experiments.run_generate_report
python -m MetaSleepGuard.experiments.run_real_openbci_reports
python -m MetaSleepGuard.tests.run_smoke_tests
```

After installing test dependencies:

```powershell
pytest MetaSleepGuard/tests -q
```

When real Sleep-EDF or ISRUC paths are not configured, users should configure local public-data paths before running formal evaluation. Demonstration-only checks are separated from submitted accuracy evidence.

## Main Modules

- `metabci_integration`: MetaBCI component discovery plus BrainFlow, Brainstim, and Brainda adapters.
- `datasets/public_sleep`: Sleep-EDF and ISRUC loaders, canonical labels `W/N1/N2/N3/REM`, and Brainda-aligned public-data adapters.
- `preprocessing`: channel selection, 250 Hz resampling, bandpass, 50 Hz notch, 30 s epochs, and subject-level splits.
- `features`: time statistics, bandpower, relative power, Hjorth, Shannon entropy, Petrosian FD, two-channel correlation, and causal context.
- `models`: sleep-staging baseline models with sklearn fallback, evaluation, cross-dataset testing, and online inference.
- `quality`: 30 s window quality audit with line noise, drift, saturation, flatline, motion artifact, dropout, abnormal amplitude, and bad-channel rules.
- `rejection`: calibration metrics, Brier score, ECE, coverage-risk, and active rejection.
- `realtime`: OpenBCI GUI file replay, BrainFlow Cyton access, ring buffer, raw save, and 30 s runtime pipeline.
- `brainstim_task`: prompts, countdown, LSL markers, and CSV logs for calibration.
- `visualization` and `reports`: waveform, spectrum, hypnogram, quality, confidence/rejection plots, and HTML reports.

## Real Data Commands

```powershell
python -m MetaSleepGuard.experiments.run_prepare_data --dataset sleep-edf --root D:\data\SleepEDF
python -m MetaSleepGuard.experiments.run_prepare_data --dataset isruc --root D:\data\ISRUC
python -m MetaSleepGuard.experiments.run_train --dataset sleep-edf --root D:\data\SleepEDF --task 5class
python -m MetaSleepGuard.experiments.run_cross_dataset --sleep-edf-root D:\data\SleepEDF --isruc-root D:\data\ISRUC --task 5class
python -m MetaSleepGuard.experiments.run_bdf_fif_audit --input-dir D:\data\boruikang
python -m MetaSleepGuard.experiments.run_openbci_file_replay --file D:\data\openbci\record.csv --model outputs\metasleepguard_outputs\models\sleep-edf_5class_baseline.joblib
python -m MetaSleepGuard.experiments.run_openbci_realtime --serial-port COM3 --model outputs\metasleepguard_outputs\models\sleep-edf_5class_baseline.joblib
```

## License

Project code is organized for GPL-2.0. MetaBCI-related dependencies and examples retain their own license notices.
