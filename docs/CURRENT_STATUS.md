# Current Status and Open Items

## Completed

- Project structure for datasets, preprocessing, features, models, quality, rejection, realtime, Brainstim, visualization, reports, experiments, tests, and docs.
- Sleep-EDF and ISRUC loaders with canonical label mapping and synthetic smoke records.
- Real Sleep-EDF validation covers 30 subjects and 32,781 valid 30-second epochs.
- Fixed-fold ablation evidence covers single/dual channel, causal context, probability calibration, and trusted rejection.
- ISRUC has interface/structure tests; real ISRUC and cross-dataset metrics are not current evidence.
- Standard preprocessing and 30-second epoching.
- Baseline feature extraction and causal context using only the previous two epochs.
- XGBoost baseline with sklearn fallback, subject-level split, full metrics, and bidirectional cross-dataset evaluation.
- Quality audit with eight artifact flags.
- Held-out-subject probability calibration, ECE/Brier metrics, active rejection, and coverage-risk evaluation integrated into training/evaluation.
- BDF/FIF audit with CSV and representative waveform/spectrum plot generation.
- OpenBCI file replay, BrainFlow Cyton wrapper, microvolt-to-volt normalization, synthetic stream, ring buffer, streaming raw CSV save, and runtime pipeline.
- Brainstim/PsychoPy calibration task with Chinese prompts, countdown, LSL marker helper, and CSV logging.
- Auto-refreshing waveform/spectrum/quality/stage dashboard and per-record HTML report generator.
- One-command scripts and tests.
- Timestamped per-record output directories prevent later runs from overwriting raw data, dashboards, or reports.
- Synthetic/public provenance is included in model metadata, terminal output, and cross-dataset metric JSON.

## Open Items

- Run real ISRUC and bidirectional cross-dataset evaluation once ISRUC files are available.
- Run real Boruikang BDF/FIF audit once files are placed under `datasets/boruikang_files`.
- Run real OpenBCI Cyton acquisition on the configured serial port.
- Replace fallback RandomForest with XGBoost by installing `xgboost` in the `metabci` environment if missing.
- Add deep learning models only after the baseline system is validated.
