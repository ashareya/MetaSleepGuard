# Current Status and Open Items

## Completed

- Project structure for datasets, preprocessing, features, models, quality, rejection, realtime, Brainstim, visualization, reports, experiments, tests, and docs.
- Sleep-EDF and ISRUC loaders with canonical label mapping and synthetic smoke records.
- Real Sleep-EDF validation covers 30 subjects and 32,781 valid 30-second epochs.
- Fixed-fold ablation evidence covers single/dual channel, causal context, probability calibration, and trusted rejection.
- Real ISRUC-Sleep validation uses 15 Subgroup-I subjects from the pinned NEMAR `nm000111 v1.0.1` release. The verified RandomForest-fallback run contains 12,661 valid epochs, five subject-grouped folds, and bidirectional cross-dataset metrics with zero train/test subject overlap.
- A forced XGBoost 3.1.3 run uses the same 15 ISRUC subjects, preprocessing, features, folds, and base seed. RandomForest and XGBoost results remain separate because neither model dominates every direction.
- Standard preprocessing and 30-second epoching.
- Baseline feature extraction and causal context using only the previous two epochs.
- XGBoost baseline with sklearn fallback, subject-level split, full metrics, and bidirectional cross-dataset evaluation.
- Quality audit with eight artifact flags.
- Held-out-subject probability calibration, ECE/Brier metrics, active rejection, and coverage-risk evaluation integrated into training/evaluation.
- BDF/FIF audit with CSV and representative waveform/spectrum plot generation.
- OpenBCI file replay, BrainFlow Cyton wrapper, microvolt-to-volt normalization, synthetic stream, ring buffer, streaming raw CSV save, and runtime pipeline.
- Verified Cyton recordings include the 10-minute quality-calibration run and the 60-minute continuous run; they support acquisition, integrity, quality, rejection, and reporting evidence, not staging accuracy.
- Brainstim/PsychoPy calibration task with Chinese prompts, countdown, LSL marker helper, and CSV logging.
- Auto-refreshing waveform/spectrum/quality/stage dashboard and per-record HTML report generator.
- One-command scripts and tests.
- Timestamped per-record output directories prevent later runs from overwriting raw data, dashboards, or reports.
- Synthetic/public provenance is included in model metadata, terminal output, and cross-dataset metric JSON.

## Open Items

- Keep the verified XGBoost and RandomForest ISRUC runs separate; neither model dominates every cross-dataset direction.
- Run real Boruikang BDF/FIF audit once files are placed under `datasets/boruikang_files`.
- Add deep learning models only after the baseline system is validated.
