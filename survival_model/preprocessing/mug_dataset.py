"""Patient table of the MUG export, in the layout of ``preprocessing.clinical_dataset``.

The MUG CSV already has one row per patient and the overall-survival outcome, so this
step parses the values into the MMCI vocabulary, fills unknown M from the stage group,
moves the age to the diagnosis and leaves out patients flagged for review. Columns
MUG does not record (site, markers, cause of death, ...) stay empty.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import hydra
import mlflow
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from preprocessing import value_parsers as vp
from preprocessing.clinical_dataset import (
    OUTPUT_COLUMNS,
    main_stage,
    median_follow_up_years,
    parse_columns,
)

log = logging.getLogger(__name__)


DATE_COLUMNS = ["examination_date", "recorded_first_diagnosis_date"]
REVIEW_FLAGS = [
    "stage_with_unknown_m",
    "age_missing",
    "category_review_flag",
    "staging_review_flag",
    "clinical_context_review_flag",
]
# Raw values of these columns are never echoed into the parse report
_PRIVATE_COLUMNS = frozenset({*DATE_COLUMNS, "patient_id"})

PARSERS = {
    "patient_id": vp.as_text,
    "age_at_examination": float,
    "sex": vp.parse_sex,
    "stage_recorded": vp.parse_stage,
    "stage_major_recorded": vp.parse_stage,
    "t_category": vp.parse_t,
    "n_category": vp.parse_n,
    "m_category": vp.parse_m,
    "grade_category": vp.parse_grade,
    "r_category": vp.parse_r,
    "l_category": vp.parse_l,
    "v_category": vp.parse_v,
    **dict.fromkeys(DATE_COLUMNS, vp.parse_date),
    "os_time_days": float,
    "os_event": vp.parse_flag,
    **dict.fromkeys(REVIEW_FLAGS, vp.parse_flag),
}

# Columns MUG has on top of the MMCI layout
EXTRA_COLUMNS = ["r", "l", "v", "examination_delay_days"]

# Every UICC stage group fixes M: IV requires M1, 0-III require M0
_M_OF_STAGE = {"0": "M0", "I": "M0", "II": "M0", "III": "M0", "IV": "M1"}


def stage_group(major: pd.Series, recorded: pd.Series) -> pd.Series:
    """Stage group (0 / I / II / III / IV); from the substaged stage when the group is missing."""
    from_recorded = recorded.map(main_stage)
    return major.where(major.notna(), from_recorded)


def fill_m(m: pd.Series, stage: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """Unknown M from the stage group; a recorded M is kept even when it contradicts the stage."""
    from_stage = stage.map(_M_OF_STAGE)
    filled = m.where(m.notna(), from_stage)
    m_group = m.map(lambda v: v[:2] if isinstance(v, str) else None)
    stats = {
        "n_m_from_stage": int((m.isna() & from_stage.notna()).sum()),
        "n_m_contradicts_stage": int((m.notna() & from_stage.notna() & (m_group != from_stage)).sum()),
    }
    return filled, stats


def examination_delay_stats(delay: pd.Series) -> dict[str, float]:
    """How long after the first diagnosis the examination (TNM, age) was recorded."""
    known = delay.dropna()
    if known.empty:
        return {}
    return {
        "examination_delay_days/median": float(known.median()),
        "examination_delay_days/p90": float(known.quantile(0.9)),
        "examination_delay_days/max": float(known.max()),
        "n_examination_before_diagnosis": int((known < 0).sum()),
    }


def era_metrics(patients: pd.DataFrame) -> dict[str, float]:
    """Patients, deaths and 5-year overall survival per decade of diagnosis.

    MUG spans decades in which surgery, chemotherapy and the TNM editions changed, so
    survival of early and recent patients may differ at the same stage.
    """
    years = patients["diagnosis_year"].dropna()
    if years.empty:
        return {}
    metrics = {"diagnosis_year/min": float(years.min()), "diagnosis_year/max": float(years.max())}
    decade = (patients["diagnosis_year"] // 10 * 10).dropna().astype(int)
    for start, group in patients.loc[decade.index].groupby(decade):
        kmf = KaplanMeierFitter().fit(group["time_days"] / 365.25, group["os_event"])
        metrics[f"era/{start}s/n_patients"] = len(group)
        metrics[f"era/{start}s/n_os_events"] = int(group["os_event"].sum())
        metrics[f"era/{start}s/os_5y"] = float(kmf.predict(5))
    return metrics


def build_patients(
    parsed: pd.DataFrame,
    exclude_flags: list[str],
    max_examination_delay_days: float | None,
    min_diagnosis_year: int | None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    stage = stage_group(parsed["stage_major_recorded"], parsed["stage_recorded"])
    m, m_stats = fill_m(parsed["m_category"], stage)

    examination = pd.to_datetime(parsed["examination_date"])
    diagnosis = pd.to_datetime(parsed["recorded_first_diagnosis_date"])
    delay = (examination - diagnosis).dt.days
    age = parsed["age_at_examination"].astype(float) - (delay / 365.25).fillna(0)
    plausible_age = age.between(15, 110)

    patients = pd.DataFrame({
        "patient_id": parsed["patient_id"],
        "n_cases": 1,
        "n_records": 1,
        "sex": parsed["sex"],
        "age": age.where(plausible_age),
        "diagnosis_year": diagnosis.dt.year,
        "grade": parsed["grade_category"],
        "pt": parsed["t_category"],
        "pn": parsed["n_category"],
        "pn_source": pd.Series("report", index=parsed.index).where(parsed["n_category"].notna()),
        "m": m,
        "clinical_stage": parsed["stage_recorded"],
        "stage_group": stage,
        "stage_source": pd.Series("reported", index=parsed.index).where(stage.notna()),
        "time_days": parsed["os_time_days"].astype(float),
        "os_event": parsed["os_event"],
        "r": parsed["r_category"],
        "l": parsed["l_category"],
        "v": parsed["v_category"],
        "examination_delay_days": delay,
    }).reindex(columns=[*OUTPUT_COLUMNS, *EXTRA_COLUMNS])

    flagged = pd.Series(False, index=parsed.index)
    stats: dict[str, float] = {**m_stats, **examination_delay_stats(delay)}
    for flag in exclude_flags:
        hit = parsed[flag].eq(True)
        stats[f"excluded_{flag}"] = int((hit & ~flagged).sum())
        flagged |= hit

    early = pd.Series(False, index=parsed.index)
    if min_diagnosis_year is not None:
        early = patients["diagnosis_year"] < min_diagnosis_year
    late = pd.Series(False, index=parsed.index)
    if max_examination_delay_days is not None:
        late = delay > max_examination_delay_days
    no_outcome = patients["time_days"].isna() | patients["os_event"].isna()
    negative = patients["time_days"] < 0
    excluded = flagged.copy()
    for name, mask in [
        ("diagnosed_before_min_year", early),
        ("late_examination", late),
        ("no_outcome", no_outcome),
        ("negative_follow_up", negative),
    ]:
        stats[f"excluded_{name}"] = int((mask & ~excluded).sum())
        excluded |= mask
    stats["n_implausible_age"] = int((age.notna() & ~plausible_age).sum())
    patients = patients[~excluded].copy()
    patients["os_event"] = patients["os_event"].astype(bool)
    patients["time_days"] = patients["time_days"].clip(lower=1)  # same-day deaths
    return patients.reset_index(drop=True), stats


def dataset_metrics(raw: pd.DataFrame, patients: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {
        "n_rows": len(raw),
        "n_patients": len(patients),
        "n_os_events": int(patients["os_event"].sum()),
        "median_follow_up_years": median_follow_up_years(patients),
    }
    for column in ["sex", "age", "grade", "pt", "pn", "m", "stage_group", "r", "l", "v"]:
        metrics[f"missing/{column}"] = float(patients[column].isna().mean())
    return metrics


# ── entrypoint ─────────────────────────────────────────────────────────────────

@with_cli_args(["+preprocessing=mug_dataset"])
@hydra.main(
    config_path="../configs",
    config_name="preprocessing",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    raw = pd.read_csv(config.dataset.path, dtype=object, keep_default_na=False)
    missing = [c for c in PARSERS if c not in raw.columns]
    if missing:
        raise ValueError(f"Columns {missing} not found in {config.dataset.path}")

    parsed, parse_report = parse_columns(raw, PARSERS, _PRIVATE_COLUMNS)
    max_delay: Any = config.dataset.max_examination_delay_days
    min_year: Any = config.dataset.min_diagnosis_year
    patients, stats = build_patients(
        parsed,
        list(config.dataset.exclude_flags),
        None if max_delay is None else float(max_delay),
        None if min_year is None else int(min_year),
    )

    for column, entry in parse_report.items():
        if entry.get("n_unparsed"):
            log.warning("%s: %d values not understood", column, entry["n_unparsed"])

    mlflow.log_metrics({
        **dataset_metrics(raw, patients),
        **era_metrics(patients),
        **{k: v for k, v in stats.items() if not np.isnan(v)},
    })

    with tempfile.TemporaryDirectory() as tmp_dir:
        patients.to_parquet(Path(tmp_dir) / "patients.parquet", index=False)
        (Path(tmp_dir) / "parse_report.json").write_text(json.dumps(parse_report, indent=2, ensure_ascii=False))
        logger.log_artifacts(tmp_dir)

    mlflow.log_input(
        mlflow.data.from_pandas(patients, name="patients"),
        context="patients",
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
