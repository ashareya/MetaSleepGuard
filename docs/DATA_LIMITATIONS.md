# Data Limitations

- Sleep-EDF and ISRUC-Sleep are the only intended sources for expert-labeled public sleep staging accuracy in this project.
- All train/test splits must be subject-level. Random epoch-level splitting is prohibited because it leaks subject information.
- Boruikang BDF/FIF files are used for real EEG compatibility, quality audit, replay, and reporting. They are not expert sleep-staging ground truth unless separate 30-second expert labels exist.
- OpenBCI Cyton recordings are used for real-time acquisition, windowed inference, quality detection, trusted rejection, and reporting. Without PSG or expert labels, they do not validate five-class sleep accuracy.
- Synthetic demo data are only for smoke tests and integration demos. Reports and README text must identify synthetic outputs as placeholders.

