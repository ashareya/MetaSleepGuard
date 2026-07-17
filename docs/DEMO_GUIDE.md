# Demo Video Script

1. Show the project structure from the repository root.
2. Show the Sleep-EDF public-data baseline workflow and explain that public expert-labeled sleep data are used for sleep-staging accuracy validation.
3. Show the quality audit module and the artifact-quality indicators.
4. Show OpenBCI file replay with a real OpenBCI GUI recording file:

       python -m MetaSleepGuard.experiments.run_openbci_file_replay --file .\data\openbci\record.csv

5. Show Brainstim calibration dry run:

       python -m MetaSleepGuard.experiments.run_brainstim_calibration --dry-run --no-psychopy

6. Generate the report:

       python -m MetaSleepGuard.experiments.run_generate_report

7. Explain the limitation: public data validate staging accuracy; OpenBCI validates the acquisition and quality-audit system path; no PSG means no clinical accuracy claim.
