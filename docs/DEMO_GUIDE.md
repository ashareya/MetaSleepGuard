# Demo Video Script

1. Show the project structure under `MetaSleepGuard/`.
2. Run a synthetic public-data training command and point out that it is a smoke test if real data are absent.
3. Show the quality audit module and the eight artifact flags.
4. Run synthetic OpenBCI real-time mode:

```powershell
python -m MetaSleepGuard.experiments.run_openbci_file_replay --file D:\data\openbci\record.csv
```

5. Show Brainstim calibration dry run:

```powershell
python -m MetaSleepGuard.experiments.run_brainstim_calibration --dry-run --no-psychopy
```

6. Generate the report:

```powershell
python -m MetaSleepGuard.experiments.run_generate_report
```

7. Explain the limitation: public data validate staging accuracy; OpenBCI validates the real-time system path; no PSG means no clinical accuracy claim.

