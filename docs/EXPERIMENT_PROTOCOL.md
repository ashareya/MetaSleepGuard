# Experiment Protocol

## 1. 10-minute Alpha and Artifact Test

Purpose:

- Verify real EEG acquisition.
- Verify eye-open/eye-closed alpha change when using occipital channels.
- Verify frontal blink/muscle/cable artifacts when using portable frontal channels.
- Verify Brainstim/PsychoPy event markers and quality detection.

Recommended electrodes:

- O1/O2 are best for alpha rhythm.
- Fp1/Fp2 are convenient for demos and naps, but alpha may be less obvious.

Default timeline:

- 0-1 min: eyes open adaptation.
- 1-3 min: eyes open rest.
- 3-5 min: eyes closed rest.
- 5-7 min: eyes open rest.
- 7-9 min: eyes closed rest.
- 9:00-9:15: blink.
- 9:15-9:30: clench teeth.
- 9:30-9:45: turn head.
- 9:45-10:00: move electrode cable.

Run:

```powershell
conda activate metabci_stim
python -m MetaSleepGuard.experiments.run_brainstim_calibration
```

## 2. 60-minute Nap or Eyes-Closed Rest

Purpose:

- Verify continuous OpenBCI acquisition.
- Verify 30-second sliding windows.
- Verify signal quality audit.
- Verify trusted rejection.
- Verify report generation.

This test is not a sleep-staging accuracy validation unless PSG/expert 30-second labels are available. If the subject does not clearly sleep, the report must state:

> 本次实验为午睡场景连续采集测试，被试未确认进入睡眠，主要用于验证 OpenBCI 实时采集、信号质量评估、滑窗推理和报告生成流程，不作为睡眠分期准确率验证依据。

