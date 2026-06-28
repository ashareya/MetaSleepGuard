# 眠卫 MetaSleep-Guard

MetaSleep-Guard is a MetaBCI-oriented two-channel EEG system for trusted sleep staging and signal-quality guarding. It is organized around Brainda-style public-data modeling, BrainFlow/OpenBCI runtime acquisition, and Brainstim/PsychoPy calibration.

## Quick Start

From the repository root:

```powershell
python -m MetaSleepGuard.experiments.run_train --dataset sleep-edf --task 5class
python -m MetaSleepGuard.experiments.run_eval --dataset sleep-edf --task 5class
python -m MetaSleepGuard.experiments.run_cross_dataset --task 5class
python -m MetaSleepGuard.experiments.run_openbci_file_replay --file D:\data\openbci\record.csv
python -m MetaSleepGuard.experiments.run_generate_report
python -m MetaSleepGuard.experiments.run_real_openbci_reports
python -m MetaSleepGuard.tests.run_smoke_tests
# After installing test dependencies:
pytest MetaSleepGuard/tests -q
```

When real Sleep-EDF or ISRUC paths are not configured, users should configure local public-data paths before running formal evaluation. Demonstration-only checks are separated from submitted accuracy evidence.

## Main Modules

- `datasets/public_sleep`: Sleep-EDF and ISRUC loaders, canonical labels `W/N1/N2/N3/REM`, and synthetic smoke records.
- `preprocessing`: channel selection, 250 Hz resampling, bandpass, 50 Hz notch, 30 s epochs, subject-level splits.
- `features`: time statistics, bandpower, relative power, Hjorth, Shannon entropy, Petrosian FD, two-channel correlation, causal context.
- `models`: XGBoost baseline with sklearn fallback, evaluation, cross-dataset testing, online inference.
- `quality`: 30 s window quality audit with line noise, drift, saturation, flatline, motion artifact, dropout, abnormal amplitude, bad-channel rules.
- `rejection`: calibration metrics, Brier score, ECE, coverage-risk, active rejection.
- `realtime`: OpenBCI GUI file replay, BrainFlow Cyton access, ring buffer, raw save, 30 s runtime pipeline.
- `brainstim_task`: Chinese prompts, countdown, LSL markers, and CSV logs for alpha/artifact calibration.
- `visualization` and `reports`: waveform, spectrum, alpha, hypnogram, quality, confidence/rejection plots, and HTML reports.

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
