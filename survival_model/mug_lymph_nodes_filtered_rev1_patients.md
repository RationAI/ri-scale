# MUG lymph nodes — CSV attributes

## Contents

[mug_lymph_nodes_filtered_rev1_patients.csv](mug_lymph_nodes_filtered_rev1_patients.csv) contains **4,492 rows and 25 columns**:
one row per patient, with demographics, staging, dates, overall-survival outcomes
and review flags. `patient_id` is the only identifier.

## Attributes

Dates use `YYYY-MM-DD`. Blank numeric values mean unavailable; a blank death date
means no death is recorded. Categorical `unknown` means missing or unspecified;
`unresolved` means the recorded value is ambiguous or cannot be assigned a clear
category. Stage and TNM values are recorded categories, including historical
suffixes, and may require review.

| Attribute | Meaning |
| --- | --- |
| `patient_id` | Unique pseudonymous patient label, e.g. patient_1. |
| `age_at_examination` | Age in years at the examination; blank if unavailable. |
| `sex` | Recorded sex: male, female or unknown. |
| `stage_recorded` | Recorded overall tumour stage, including substage where available, e.g. IIA or IIIB. |
| `stage_major_recorded` | Overall stage group without substage: 0, I, II, III or IV. |
| `t_category` | Recorded T category: extent of the primary tumour, e.g. T3 or T4B. |
| `n_category` | Recorded N category: regional lymph-node involvement, e.g. N0 or N1B. |
| `m_category` | Recorded M category: distant metastasis status, e.g. M0, M1 or M1B. Unknown does not mean M0. |
| `grade_category` | Histological tumour grade: G1–G4. |
| `r_category` | Recorded residual-tumour category: R0–R2. |
| `l_category` | Recorded lymphatic-invasion category: L0 or L1. |
| `v_category` | Recorded venous-invasion category: V0–V2. |
| `examination_date` | Date of the examination associated with this patient row. |
| `recorded_first_diagnosis_date` | Recorded first-diagnosis date; the time origin for overall survival. |
| `diagnosis_year` | Calendar year of the recorded first diagnosis. |
| `os_time_days` | Days from first diagnosis to death, or to the follow-up cutoff for censored patients. |
| `os_event` | 1 = observed death; 0 = recorded alive at the follow-up cutoff (right-censored). |
| `death_date` | Recorded death date; blank for censored patients. |
| `followup_cutoff_date` | Recorded follow-up cutoff; the survival endpoint for censored patients. |
| `endpoint_date` | Death date when os_event = 1; follow-up cutoff when os_event = 0. |

Use `os_time_days` together with `os_event` to describe overall survival. Censoring
means survival beyond the follow-up cutoff is unknown; it does not mean the patient
is alive today. The endpoint is overall survival, not cancer-specific survival.

## Review and quality flags

All flags use **1 = condition present or review needed**, **0 = flag not raised**.

| Flag | Meaning when 1 | Patients |
| --- | --- | ---: |
| `stage_with_unknown_m` | An overall stage is recorded, but the M category is unknown. | 3,910 |
| `age_missing` | Age at examination is unavailable. | 12 |
| `category_review_flag` | At least one categorical value is unresolved because it is ambiguous or unsupported. Missing values alone do not trigger this flag. | 19 |
| `staging_review_flag` | Recorded stage needs review because of liver-metastasis or peritoneal/omental disease context in the clinical report. | 12 |
| `clinical_context_review_flag` | Clinical context needs review: possible or documented recurrence, prior treatment, multiple tumours, unusual primary site/histology, or uncertain tumour origin. | 252 |

Flags can overlap. A review flag does not by itself establish an error, and a zero
flag does not guarantee complete or clinically verified information.
