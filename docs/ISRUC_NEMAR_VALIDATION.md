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

## Verified RandomForest run

Command:

```powershell
.\run.ps1 -Task isruc-validation -MaxSubjects 15
```

This run completed on 2026-07-17 with 15 subjects and 12,661 valid 30-second epochs. All internal folds use `GroupKFold(n_splits=5, group=subject_id)`. No internal fold or cross-dataset direction contains train/test subject overlap. XGBoost was unavailable for this run, so every metric below is explicitly a `sklearn_random_forest_fallback` result.

| Task | Evaluation | Accuracy | Macro-F1 | Weighted-F1 | Kappa |
|---|---|---:|---:|---:|---:|
| 3 class | ISRUC internal five-fold | 0.729168 | 0.634353 | 0.724579 | 0.513172 |
| 3 class | Sleep-EDF to ISRUC | 0.735487 | 0.532871 | 0.696395 | 0.509721 |
| 3 class | ISRUC to Sleep-EDF | 0.550038 | 0.376234 | 0.554287 | 0.252158 |
| 5 class | ISRUC internal five-fold | 0.647500 | 0.551406 | 0.625683 | 0.532057 |
| 5 class | Sleep-EDF to ISRUC | 0.382039 | 0.254568 | 0.331548 | 0.201084 |
| 5 class | ISRUC to Sleep-EDF | 0.591714 | 0.335364 | 0.613274 | 0.309591 |

## Verified XGBoost run

XGBoost `3.1.3` was installed into the `metabci` environment with `pip install --no-deps`, leaving NumPy `2.2.6` and scikit-learn `1.7.2` unchanged. XGBoost is a third-party classifier used by this compatible extension; it is not a MetaBCI official algorithm. The experiment was forced to use it with:

```powershell
.\run.ps1 -Task isruc-validation -MaxSubjects 15 -RequireModel xgboost
```

The forced run completed on 2026-07-17 with the same 15 subjects, 12,661 epochs, preprocessing, features, mappings, folds, and base random seed. Each fold records XGBoost version, parameters, and its derived seed. Silent RandomForest fallback is disabled by `-RequireModel xgboost`.

| Task | Evaluation | Accuracy | Macro-F1 | Weighted-F1 | Kappa |
|---|---|---:|---:|---:|---:|
| 3 class | ISRUC internal five-fold | 0.743701 | 0.644365 | 0.740055 | 0.538395 |
| 3 class | Sleep-EDF to ISRUC | 0.429113 | 0.416684 | 0.413979 | 0.216173 |
| 3 class | ISRUC to Sleep-EDF | 0.680469 | 0.461599 | 0.683407 | 0.410578 |
| 5 class | ISRUC internal five-fold | 0.645921 | 0.564890 | 0.635709 | 0.535884 |
| 5 class | Sleep-EDF to ISRUC | 0.404470 | 0.294258 | 0.360312 | 0.245659 |
| 5 class | ISRUC to Sleep-EDF | 0.725967 | 0.404728 | 0.731933 | 0.480443 |

## Same-protocol comparison

| Task | Evaluation | XGBoost Accuracy | RandomForest Accuracy | Difference |
|---|---|---:|---:|---:|
| 3 class | ISRUC internal five-fold | 0.743701 | 0.729168 | +0.014533 |
| 3 class | Sleep-EDF to ISRUC | 0.429113 | 0.735487 | -0.306374 |
| 3 class | ISRUC to Sleep-EDF | 0.680469 | 0.550038 | +0.130432 |
| 5 class | ISRUC internal five-fold | 0.645921 | 0.647500 | -0.001580 |
| 5 class | Sleep-EDF to ISRUC | 0.404470 | 0.382039 | +0.022431 |
| 5 class | ISRUC to Sleep-EDF | 0.725967 | 0.591714 | +0.134254 |

Neither model dominates every direction. In particular, XGBoost improves both ISRUC-to-Sleep-EDF directions but is substantially worse on the three-class Sleep-EDF-to-ISRUC direction. Both complete result sets must therefore remain visible.

The experiment writes `status.json`, `data_manifest.json`, full per-class JSON/CSV, confusion matrices, and the summary into a unique timestamped directory under `outputs/metasleepguard_outputs/isruc_validation/`. A result is valid only when `status.json` contains `status=complete` and `metrics_generated=true`.
