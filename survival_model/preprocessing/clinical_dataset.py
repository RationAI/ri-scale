import hashlib
import json
import logging
import re
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import hydra
import mlflow
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from preprocessing import value_parsers as vp
from preprocessing.data_source import ExcelDataSource

log = logging.getLogger(__name__)


# ── record parsing ─────────────────────────────────────────────────────────────

DATE_COLUMNS = ["procedure_date", "birth_date", "death_date", "last_visit_date", "diagnosis_date"]
COUNT_COLUMNS = ["sn_examined", "sn_positive", "ln_examined", "ln_positive"]
DEATH_CAUSES = ["death_cause_underlying", "death_cause_other", "death_cause_immediate"]
# Raw values of these columns are never echoed into the parse report
_PRIVATE_COLUMNS = frozenset({*DATE_COLUMNS, "biopsy_id", "sample_id"})

PARSERS: dict[str, Callable[[Any], Any]] = {
    "biopsy_id": vp.normalize_case_id,
    "sample_id": vp.as_text,
    "topography": vp.parse_icd,
    "morphology": vp.parse_morphology,
    "grade": vp.parse_grade,
    "diagnosis": vp.parse_icd,
    "y_prefix": vp.parse_flag,
    "r_prefix": vp.parse_flag,
    "pt": vp.parse_t,
    "pn": vp.parse_n,
    "pm": vp.parse_m,
    "cm": vp.parse_m,
    **dict.fromkeys(COUNT_COLUMNS, vp.parse_count),
    "sex": vp.parse_sex,
    **dict.fromkeys(DATE_COLUMNS, vp.parse_date),
    **dict.fromkeys(DEATH_CAUSES, vp.parse_icd),
    "comorbidity": vp.parse_icd,
    "clinical_stage": vp.parse_stage,
    "mmr": vp.parse_mmr,
    "kras": vp.parse_mutation,
    "nras": vp.parse_mutation,
    "braf": vp.parse_mutation,
}


def _parse_column(values: pd.Series, parser: Callable[[Any], Any]) -> tuple[list, Counter]:
    parsed: list = []
    unparsed: Counter = Counter()
    for value in values:
        if vp.is_missing(value):
            parsed.append(None)
            continue
        try:
            parsed.append(parser(value))
        except ValueError:
            parsed.append(None)
            unparsed[vp.as_text(value)] += 1
    return parsed, unparsed


def parse_records(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Parse every cell into its canonical value; returns the records and a parse report."""
    records = pd.DataFrame(index=raw.index)
    report: dict[str, dict] = {}
    for column, parser in PARSERS.items():
        if column not in raw:
            records[column] = None
            report[column] = {"present": False}
            continue
        parsed, unparsed = _parse_column(raw[column], parser)
        records[column] = parsed
        report[column] = {
            "present": True,
            "n_filled": int((~raw[column].map(vp.is_missing)).sum()),
            "n_unparsed": sum(unparsed.values()),
            "unparsed_values": {} if column in _PRIVATE_COLUMNS else dict(unparsed.most_common(50)),
        }

    for column in DATE_COLUMNS:
        records[column] = pd.to_datetime(records[column])
    for column in COUNT_COLUMNS:
        records[column] = records[column].astype(float)

    # y / r may be written in front of the TNM values instead of in their own columns
    for flag, letter in (("y_prefix", "y"), ("r_prefix", "r")):
        embedded = pd.Series(False, index=raw.index)
        for column, category in (("pt", "T"), ("pn", "N")):
            if column in raw:
                embedded |= raw[column].map(
                    lambda v, c=category, ch=letter: not vp.is_missing(v) and ch in vp.tnm_prefix(v, c)
                )
        records[flag] = records[flag].map(bool) | embedded

    return records.rename(columns={"biopsy_id": "case_id"}), report


NODE_PAIRS = [("sn_positive", "sn_examined"), ("ln_positive", "ln_examined")]


def clean_node_counts(records: pd.DataFrame) -> dict[str, int]:
    """Blank sentinel counts of a biopsy that was not done and count inconsistent pairs.

    The export writes 0 / 0 when no sentinel biopsy was done, which is "not done"
    rather than "0 nodes".
    """
    not_done = records["sn_examined"] == 0
    records.loc[not_done, ["sn_positive", "sn_examined"]] = np.nan
    stats = {"n_sentinel_not_done": int(not_done.sum())}
    for positive, examined in NODE_PAIRS:
        stats[f"n_{positive}_exceeds_examined"] = int((records[positive] > records[examined]).sum())
        if stats[f"n_{positive}_exceeds_examined"]:
            log.warning("%d records with %s > %s", stats[f"n_{positive}_exceeds_examined"], positive, examined)
    return stats


def crc_mask(records: pd.DataFrame, include: list[str], exclude: list[str]) -> pd.Series:
    """Rows of colorectal primaries; rows with an unknown site are kept."""
    code = records["topography"].fillna(records["diagnosis"])

    def keep(c: str | None) -> bool:
        if not isinstance(c, str):
            return True
        if c.startswith(tuple(exclude)):
            return False
        return c.startswith(tuple(include))

    return code.map(keep).astype(bool)


# ── patient aggregation ────────────────────────────────────────────────────────

TUMOR_FIELDS = [
    "topography", "diagnosis", "morphology", "grade", "pt", "pn", "pm", "cm",
    "sn_examined", "sn_positive", "ln_examined", "ln_positive", "clinical_stage",
    "y_prefix", "r_prefix",
]
# Marker -> value that wins when the samples of one patient disagree
MARKERS = {"mmr": "dMMR", "kras": "mut", "nras": "mut", "braf": "mut"}


def _patient_keys(
    records: pd.DataFrame, key_columns: list[str], nullable_columns: list[str]
) -> pd.Series:
    """``key_columns`` joined (an empty nullable column is a value too, e.g. "alive").

    Rows missing a non-nullable key column fall back to their case (or row).
    """
    key = records[key_columns].astype(str).agg("|".join, axis=1)
    required = [c for c in key_columns if c not in nullable_columns]
    row_ids = pd.Series("row" + records.index.astype(str), index=records.index)
    fallback = "case|" + records["case_id"].where(records["case_id"].notna(), row_ids).astype(str)
    return key.where(records[required].notna().all(axis=1), fallback)


def aggregate_patients(
    records: pd.DataFrame, key_columns: list[str], nullable_columns: list[str]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """One row per patient.

    Tumour fields come from the most complete primary-tumour record (pathological
    staging present, most fields filled, earliest), gaps are filled from the other
    records of the same patient.
    """
    records = records.assign(
        patient_key=_patient_keys(records, key_columns, nullable_columns),
        _has_pathology=records["pt"].notna() & records["pn"].notna(),
        _n_tumor_fields=records[TUMOR_FIELDS].notna().sum(axis=1),
    ).sort_values(
        ["r_prefix", "_has_pathology", "_n_tumor_fields", "diagnosis_date"],
        ascending=[True, False, False, True],
        na_position="last",
    )
    grouped = records.groupby("patient_key", sort=False)

    patients = grouped[[*TUMOR_FIELDS, "sex", "birth_date", *DEATH_CAUSES]].first()
    patients["diagnosis_date"] = grouped["diagnosis_date"].min()
    patients["procedure_date"] = grouped["procedure_date"].min()
    patients["last_visit_date"] = grouped["last_visit_date"].max()
    patients["death_date"] = grouped["death_date"].max()
    for marker, positive in MARKERS.items():
        any_positive = grouped[marker].agg(lambda s, p=positive: (s == p).any())
        patients[marker] = grouped[marker].first().where(~any_positive, positive)
    patients["case_ids"] = grouped["case_id"].agg(lambda s: ";".join(sorted(set(s.dropna()))))
    patients["n_records"] = grouped.size()
    patients["n_cases"] = grouped["case_id"].nunique()

    required = [c for c in key_columns if c not in nullable_columns]
    stats = {
        # Patients told apart only by a nullable key column (e.g. same birth date and sex)
        "n_split_by_nullable_key": int(
            (patients.groupby([patients[c].astype(str) for c in required]).size() > 1).sum()
        ) if nullable_columns else 0,
        # Several distinct last visits / primary sites hint at two people under one key
        "n_conflicting_last_visit": int((grouped["last_visit_date"].nunique() > 1).sum()),
        "n_multiple_primary_sites": int(
            (grouped["topography"].agg(lambda s: s.dropna().str[:3].nunique()) > 1).sum()
        ),
    }
    patients.index = patients.index.map(lambda k: hashlib.sha256(k.encode()).hexdigest()[:12])
    patients.index.name = "patient_id"
    return patients.reset_index(), stats


# ── derived tumour fields ──────────────────────────────────────────────────────

def _empty(df: pd.DataFrame) -> pd.Series:
    return pd.Series(None, index=df.index, dtype=object)


def _starts_with(series: pd.Series, prefix: str) -> pd.Series:
    return series.map(lambda v: isinstance(v, str) and v.startswith(prefix)).astype(bool)


def _main_stage(stage: Any) -> str | None:
    match = re.match(r"^(0|IV|I{1,3})", stage) if isinstance(stage, str) else None
    return match.group(1) if match else None


def _derive_stage(pt: Any, pn: Any, m: Any) -> str | None:
    """Main UICC stage group from TNM (identical in TNM 7 and 8); unknown M counts as M0."""
    if isinstance(m, str) and m.startswith("M1"):
        return "IV"
    if not (isinstance(pt, str) and isinstance(pn, str)):
        return None
    if pn != "N0":
        return "III"
    if pt in ("T0", "Tis"):
        return "0"
    return "II" if pt[:2] in ("T3", "T4") else "I"


def add_tumor_fields(patients: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    pm, cm = patients["pm"], patients["cm"]
    # Pathological M wins unless only the clinical one shows metastases
    patients["m"] = pm.where(pm.notna(), cm).where(
        ~_starts_with(cm, "M1") | _starts_with(pm, "M1"), cm
    )

    patients["nodes_examined"] = patients[["ln_examined", "sn_examined"]].sum(axis=1, min_count=1)
    patients["nodes_positive"] = patients[["ln_positive", "sn_positive"]].sum(axis=1, min_count=1)
    from_counts = pd.Series(
        pd.cut(patients["nodes_positive"], [-0.5, 0.5, 3.5, np.inf], labels=["N0", "N1", "N2"]),
        index=patients.index,
    ).astype(object)
    from_counts = from_counts.where(patients["nodes_examined"] > 0)
    patients["pn_source"] = (
        _empty(patients).mask(from_counts.notna(), "counts").mask(patients["pn"].notna(), "report")
    )
    patients["pn"] = patients["pn"].where(patients["pn"].notna(), from_counts)

    reported = patients["clinical_stage"].map(_main_stage)
    derived = pd.Series(
        [_derive_stage(*row) for row in patients[["pt", "pn", "m"]].itertuples(index=False)],
        index=patients.index,
        dtype=object,
    )
    both = reported.notna() & derived.notna()
    patients["stage_group"] = reported.where(reported.notna(), derived)
    patients["stage_source"] = (
        _empty(patients).mask(derived.notna(), "derived").mask(reported.notna(), "reported")
    )
    patients["neoadjuvant"] = patients["y_prefix"].astype(bool)

    stats = {
        "n_pn_from_counts": int((patients["pn_source"] == "counts").sum()),
        "n_stage_derived": int((patients["stage_source"] == "derived").sum()),
        "stage_agreement": float((reported[both] == derived[both]).mean()) if both.any() else float("nan"),
    }
    return patients, stats


# ── outcome ────────────────────────────────────────────────────────────────────

def add_outcome(
    patients: pd.DataFrame,
    cancer_death_codes: list[str],
    cancer_death_lines: list[str],
    administrative_censor_date: str | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    start = patients["diagnosis_date"].fillna(patients["procedure_date"])
    death = patients["death_date"]
    dead = death.notna()
    if administrative_censor_date is None:
        end_alive = patients["last_visit_date"]
    else:
        censor = pd.Timestamp(administrative_censor_date)
        dead &= death <= censor
        end_alive = pd.Series(censor, index=patients.index)
    end = death.where(dead, end_alive)

    def crc_code(code: Any) -> bool:
        return isinstance(code, str) and code.startswith(tuple(cancer_death_codes))

    # Underlying cause, or the lowest filled line of part I when it is missing
    cause = patients["death_cause_underlying"]
    for fallback in DEATH_CAUSES[1:]:
        cause = cause.fillna(patients[fallback])
    crc_in_lines = patients[cancer_death_lines].map(crc_code).any(axis=1)
    crc_in_chain = patients[DEATH_CAUSES].map(crc_code).any(axis=1)
    is_crc = cause.map(crc_code).astype(bool) | crc_in_lines

    age = (start - patients["birth_date"]).dt.days / 365.25
    plausible_age = age.between(15, 110)

    patients = patients.assign(
        time_days=(end - start).dt.days,
        os_event=dead,
        css_event=dead & is_crc,
        death_cause=cause.where(dead),
        death_cause_group=pd.Series("other", index=patients.index)
        .mask(is_crc, "crc")
        .mask(cause.isna(), "unknown")
        .where(dead, None),
        age=age.where(plausible_age),
        diagnosis_year=start.dt.year,
    )

    stats = {
        "n_start_from_procedure_date": int((patients["diagnosis_date"].isna() & start.notna()).sum()),
        "n_implausible_age": int((age.notna() & ~plausible_age).sum()),
        "n_visit_after_death": int(
            (dead & ((patients["last_visit_date"] - death).dt.days > 30)).sum()
        ),
        "n_deaths_unknown_cause": int((patients["death_cause_group"] == "unknown").sum()),
        # Deaths with a CRC code somewhere in part I that do not count as CRC deaths
        # (e.g. underlying D03.5 with C19 in lines Ia / Ic); see dataset.cancer_death_lines
        "n_crc_in_chain_not_counted": int((dead & crc_in_chain & ~is_crc).sum()),
        "excluded_no_diagnosis_date": int(start.isna().sum()),
        "excluded_no_follow_up": int((start.notna() & end.isna()).sum()),
        "excluded_negative_follow_up": int((patients["time_days"] < 0).sum()),
    }
    patients = patients[patients["time_days"] >= 0].copy()
    patients["time_days"] = patients["time_days"].clip(lower=1)  # same-day deaths
    return patients, stats


# ── dataset ────────────────────────────────────────────────────────────────────

# No dates, names or registry keys leave this step; time is relative to diagnosis.
OUTPUT_COLUMNS = [
    "patient_id", "case_ids", "n_cases", "n_records",
    "sex", "age", "diagnosis_year",
    "topography", "diagnosis", "morphology", "grade",
    "pt", "pn", "pn_source", "m", "pm", "cm", "neoadjuvant", "r_prefix",
    "nodes_examined", "nodes_positive", "ln_examined", "ln_positive", "sn_examined", "sn_positive",
    "clinical_stage", "stage_group", "stage_source",
    "mmr", "kras", "nras", "braf",
    "time_days", "os_event", "css_event", "death_cause", "death_cause_group",
]


def build_patients(
    records: pd.DataFrame, config: DictConfig
) -> tuple[pd.DataFrame, dict[str, float]]:
    patients, stats = aggregate_patients(
        records, list(config.patient_key), list(config.patient_key_nullable)
    )
    patients, tumor_stats = add_tumor_fields(patients)
    patients, outcome_stats = add_outcome(
        patients,
        list(config.cancer_death_codes),
        list(config.cancer_death_lines),
        config.administrative_censor_date,
    )
    return patients[OUTPUT_COLUMNS].reset_index(drop=True), {**stats, **tumor_stats, **outcome_stats}


def _median_follow_up_years(patients: pd.DataFrame) -> float:
    """Reverse Kaplan-Meier median follow-up."""
    kmf = KaplanMeierFitter().fit(patients["time_days"], event_observed=~patients["os_event"])
    return float(kmf.median_survival_time_) / 365.25


def dataset_metrics(records: pd.DataFrame, patients: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {
        "n_records": len(records),
        "n_cases": int(records["case_id"].nunique()),
        "n_patients": len(patients),
        "n_os_events": int(patients["os_event"].sum()),
        "n_css_events": int(patients["css_event"].sum()),
        "median_follow_up_years": _median_follow_up_years(patients),
    }
    for column in OUTPUT_COLUMNS:
        if column not in ("patient_id", "time_days", "os_event", "css_event"):
            metrics[f"missing/{column}"] = float(patients[column].isna().mean())
    return metrics


# ── entrypoint ─────────────────────────────────────────────────────────────────

@with_cli_args(["+preprocessing=clinical_dataset"])
@hydra.main(
    config_path="../configs",
    config_name="preprocessing",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    source: ExcelDataSource = hydra.utils.instantiate(config.dataset.source)
    columns = OmegaConf.to_container(config.dataset.columns, resolve=True)
    raw, mappings = source.read(columns)  # type: ignore[arg-type]

    missing = [c for c in config.dataset.required if c not in raw.columns]
    if missing:
        raise ValueError(
            f"Required columns {missing} not found; matched columns per sheet: {mappings}"
        )

    records, parse_report = parse_records(raw)
    node_stats = clean_node_counts(records)
    is_crc = crc_mask(
        records, list(config.dataset.include_topography), list(config.dataset.exclude_topography)
    )
    patients, stats = build_patients(records[is_crc], config.dataset)

    for column, entry in parse_report.items():
        if entry.get("n_unparsed"):
            log.warning("%s: %d values not understood", column, entry["n_unparsed"])

    mlflow.log_metrics({
        **dataset_metrics(records, patients),
        **{k: v for k, v in stats.items() if not pd.isna(v)},
        **node_stats,
        "excluded_non_crc_records": int((~is_crc).sum()),
    })

    with tempfile.TemporaryDirectory() as tmp_dir:
        patients.to_parquet(Path(tmp_dir) / "patients.parquet", index=False)
        (Path(tmp_dir) / "parse_report.json").write_text(json.dumps(parse_report, indent=2, ensure_ascii=False))
        (Path(tmp_dir) / "column_mapping.json").write_text(json.dumps(mappings, indent=2, ensure_ascii=False))
        logger.log_artifacts(tmp_dir)

    mlflow.log_input(
        mlflow.data.from_pandas(patients, name="patients"),
        context="patients",
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
