# Survival model

Tabular baseline for overall / cancer-specific survival of the MMCI colorectal cohort.
It is a Cox proportional-hazards regression with a random survival forest and an
elastic-net Cox as alternatives. It also includes an analysis of which clinical
variables matter. The layout mirrors `tissue_classifier`: every step is a Hydra
entrypoint that logs to MLflow.

## Pipeline

```bash
# 1. Excel sheets -> one row per patient (patients.parquet, parse_report.json, column_mapping.json)
uv run -m preprocessing.clinical_dataset +data=raw/mmci dataset.source.src_dir=/path/to/exports

# 2. Held-out test set + 5 CV folds, stratified by OS event and stage
uv run -m preprocessing.dataset_split +data=split/mmci patients_uri=mlflow-artifacts:/<exp>/<run>/artifacts/patients.parquet

# 3. Cross-validated model + feature analysis (set data.splits_uri in configs/data/datasets/mmci.yaml)
uv run -m survival_model +experiment=mmci                       # Cox, overall survival
uv run -m survival_model +experiment=mmci model=rsf             # random survival forest
uv run -m survival_model +experiment=mmci model=coxnet          # elastic-net Cox
uv run -m survival_model +experiment=mmci features=stage_only   # clinical reference: stage group alone
uv run -m survival_model +experiment=mmci endpoint=css          # colorectal-cancer-specific survival
```

`scripts/*.py` submit the same steps as kube jobs. To try everything without patient data, run
`uv run scripts/synthetic_excel.py /tmp/export/mmci.xlsx`. It writes a synthetic export with the
real header and a known ground truth: sex, KRAS and NRAS have no effect.

## Preprocessing

* Headers are matched by the regexes in `configs/data/raw/mmci.yaml` (accents and case ignored),
  so new or renamed columns only need a config change. The header row is found automatically,
  and sheets without one (legends) are skipped.
* Cell parsers live in `preprocessing/value_parsers.py`. `parse_report.json` counts the values they
  do not understand and lists what each raw value was parsed to (raw values only for categorical
  columns, never for dates or IDs). Check that report first on new data.
* The export has no patient ID, so rows are grouped by birth date + sex + death date. Within
  a patient, tumour fields come from the most complete primary-tumour record. The metrics
  `n_split_by_nullable_key` and `n_conflicting_last_visit` show how often the key is ambiguous.
* Survival time runs from diagnosis to death or to the last visit. A death counts for CSS when the
  underlying cause (falling back to lines Ic and Ia) is in `cancer_death_codes`. Some certificates
  have a non-CRC underlying cause with CRC in lines Ia / Ic. `n_crc_in_chain_not_counted` counts
  them, and `cancer_death_lines` decides whether they count as CRC deaths.
* Sentinel counts of 0 / 0 mean no sentinel biopsy and are treated as missing.
* No dates, names or registry keys leave this step. `patient_id` is a hash of the key, and
  `case_ids` (`YYYY_NNNNN`, the slide naming) is kept for joining slides later.

## Feature analysis (artifacts of a training run)

| artifact | question it answers |
|---|---|
| `feature_summary.csv` | everything below, one row per feature |
| `feature_tests.csv` | univariable and **adjusted** (drop-one likelihood-ratio, Holm-corrected) Cox tests |
| `drop_group_importance.png` | how much CV C-index is lost without the feature; ≈0 means it can be omitted |
| `cox_multivariable.csv`, `forest_plot.png` | adjusted hazard ratios (per year for age) |
| `km_curves.png` | Kaplan-Meier curves and log-rank test per categorical feature |
| `proportional_hazards_test.csv` | features whose effect changes over time (Cox assumption) |
| `cv_risk_groups.png` | separation of out-of-fold risk tertiles |

Read the adjusted test and the drop-one ΔC-index together. A feature can be significant
univariably only because of confounding: on the synthetic data KRAS looks strongly prognostic
alone, because testing was done mostly in stage IV, and nothing remains after adjustment.
The statistics use the training pool only. Keep `evaluate_test=false` while choosing features.

Outcome-derived columns (death causes, comorbidity from the death certificate) are never
features, because they exist only for patients who died. Stage group is left out of the
default feature set because it is a function of pT/pN/M.
