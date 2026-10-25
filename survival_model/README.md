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

## MUG cohort and external validation

The MUG export (`mug_lymph_nodes_filtered_rev1_patients.csv`) already has one row per patient and
overall survival, so `preprocessing.mug_dataset` only maps it to the same `patients.parquet` layout.
Option A is to develop the model on MUG, which is the larger cohort, and validate it on MMCI. That
model uses the `shared` feature set: age, sex, pT, pN, M and grade, which both institutes record.

```bash
uv run -m preprocessing.mug_dataset +data=raw/mug dataset.path=/path/to/mug_lymph_nodes_filtered_rev1_patients.csv
uv run -m preprocessing.dataset_split +data=split/mug patients_uri=mlflow-artifacts:/<exp>/<run>/artifacts/patients.parquet
uv run -m survival_model +experiment=mug data.splits_uri=<MUG split run> \
    data.external_uri=mlflow-artifacts:/<exp>/<MMCI run>/artifacts/patients.parquet
```

* M is unknown for most MUG patients, but the stage group fixes it: IV means M1 and 0-III mean M0
  (`n_m_from_stage`). A recorded M is kept even when it contradicts the stage (`n_m_contradicts_stage`).
* Age is recorded at the examination and is moved back to the first diagnosis, where survival starts.
  Patients examined more than `max_examination_delay_days` after the diagnosis are left out, as they
  had to survive until the examination.
* Patients with `clinical_context_review_flag` (recurrence, prior treatment, multiple tumours, ...)
  are left out (`exclude_flags`). R, L and V are kept in `patients.parquet` but MMCI does not have them.
* MUG covers diagnoses from 1984 to 2013 (follow-up until 2018) and is almost entirely stage II-III;
  the `era/*` metrics give 5-year survival per decade of diagnosis. `cohort.min_diagnosis_year`
  and `cohort.exclude_stage_groups` select the patients of the CV folds, the test set and the
  external set alike, e.g. `cohort.min_diagnosis_year=2000 'cohort.exclude_stage_groups=[IV]'`.
* `data.external_uri` evaluates the final model on another institute's patients
  (`external/*` metrics, `external_predictions.csv`). As with the test set, use it only once the model
  and feature set are fixed. Only overall survival is available for MUG.

`scripts/*.py` submit the same steps as kube jobs. The split and training scripts use MMCI unless
`+data=split/mug` or `+experiment=mug` is given. To try everything without patient data, run
`uv run scripts/synthetic_excel.py /tmp/export/mmci.xlsx`. It writes a synthetic MMCI export with the
real header and a known ground truth: sex, KRAS and NRAS have no effect.

## Survival from the lymph-node images (MMCI)

A second model predicts overall survival from the segmented lymph nodes (SAM3 predictions, see the lymph-node
segmentation project) plus age and sex, without the stage, TNM or markers of the Excel sheet. It runs on musica
(the slides stay there), in `image_survival/`:

1. `scripts/extract_virchow2.py` embeds every 256-px tile (0.5 um/px) that overlaps a predicted node or its 50 um margin
   with Virchow2 (class token + mean patch token, 2560-d, fp16), one `.h5` per slide. Slides are read in place and only
   the `.h5` files are written. Needs `timm`, `h5py`, `opencv`, `openslide` (install them inside the Slurm job).
2. `python -m image_survival.pool_nodes` averages the tiles of every node (core: at least half of the tile is node;
   margin: below 5 %) into `nodes.npz`. The slide name is reduced to a running index, so the LNN / LNP / M suffix, which is
   a pathologist's staging result, cannot reach the model.
3. `python -m image_survival.run` scores age + sex only, pooled embeddings (PCA + ridge Cox) and a gated-attention
   multiple-instance model with a Cox loss on the same repeated patient-level folds, and reports the gain over age + sex
   fold by fold. Hyper-parameters are fixed, not tuned on the folds. With ~200 patients and ~90 deaths a single split
   says little, hence the repeats.

`scripts/synthetic_nodes.py` writes synthetic embeddings with a known signal to test steps 2-3 without patient data.

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
