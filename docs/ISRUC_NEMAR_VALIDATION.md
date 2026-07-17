# ISRUC-Sleep NEMAR Validation

## Provenance

- Dataset: ISRUC-Sleep, Subgroup I, subjects `sub-I001` through `sub-I015`
- Distribution: NEMAR `nm000111 v1.0.1`
- NEMAR DOI: `10.82901/nemar.nm000111`
- ISRUC paper DOI: `10.1016/j.cmpb.2015.10.013`
- Labels: scorer 1; scorer 2 is retained only where present in `events.tsv`
- Download manifest: 95 selected files, 95 complete, 0 incomplete; every EDF is checked against the pinned SHA256

Raw EDF files are local dataset inputs and are not committed to Git. Recreate them with:

```powershell
.\run.ps1 -Task isruc-download -MaxSubjects 15
```

## Verified run

Command:

```powershell
.\run.ps1 -Task isruc-validation -MaxSubjects 15
```

The verified run completed on 2026-07-17 with 15 subjects and 12,661 valid 30-second epochs. All internal folds use `GroupKFold(n_splits=5, group=subject_id)`. No internal fold or cross-dataset direction contains train/test subject overlap. XGBoost was unavailable in the selected `metabci` environment, so every metric below is explicitly a `sklearn_random_forest_fallback` result.

| Task | Evaluation | Accuracy | Macro-F1 | Weighted-F1 | Kappa |
|---|---|---:|---:|---:|---:|
| 3 class | ISRUC internal five-fold | 0.729168 | 0.634353 | 0.724579 | 0.513172 |
| 3 class | Sleep-EDF to ISRUC | 0.735487 | 0.532871 | 0.696395 | 0.509721 |
| 3 class | ISRUC to Sleep-EDF | 0.550038 | 0.376234 | 0.554287 | 0.252158 |
| 5 class | ISRUC internal five-fold | 0.647500 | 0.551406 | 0.625683 | 0.532057 |
| 5 class | Sleep-EDF to ISRUC | 0.382039 | 0.254568 | 0.331548 | 0.201084 |
| 5 class | ISRUC to Sleep-EDF | 0.591714 | 0.335364 | 0.613274 | 0.309591 |

The experiment writes `status.json`, `data_manifest.json`, full per-class JSON/CSV, confusion matrices, and the summary into a unique timestamped directory under `outputs/metasleepguard_outputs/isruc_validation/`. A result is valid only when `status.json` contains `status=complete` and `metrics_generated=true`.
