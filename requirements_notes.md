# Requirements Notes

Analysis environment `metabci`:

- MetaBCI 0.2.0 with `metabci.brainda` and `metabci.brainflow`
- numpy, scipy, pandas, scikit-learn, joblib, matplotlib, PyYAML
- MNE for EDF/BDF/FIF reading
- XGBoost for the preferred baseline; if absent, the code uses a RandomForest fallback for smoke tests
- BrainFlow SDK for OpenBCI Cyton real-time acquisition

Stimulation environment `metabci_stim`:

- MetaBCI 0.2.0 with `metabci.brainstim`
- PsychoPy 2026.1.3
- pylsl

Do not install PsychoPy into the analysis environment. Use `metabci` for acquisition/analysis and `metabci_stim` for Brainstim/PsychoPy tasks.

