from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from mlflow.artifacts import download_artifacts

from survival_model.features import derive_features, make_target


@dataclass
class Fold:
    X: pd.DataFrame  # raw features (see derive_features)
    y: np.ndarray    # structured (event, time) array
    patient_id: pd.Series

    @classmethod
    def from_patients(cls, patients: pd.DataFrame, endpoint: str) -> "Fold":
        patients = patients.reset_index(drop=True)
        return cls(derive_features(patients), make_target(patients, endpoint), patients["patient_id"])

    @classmethod
    def concat(cls, folds: list["Fold"]) -> "Fold":
        return cls(
            pd.concat([f.X for f in folds], ignore_index=True),
            np.concatenate([f.y for f in folds]),
            pd.concat([f.patient_id for f in folds], ignore_index=True),
        )


def select_cohort(
    patients: pd.DataFrame, min_diagnosis_year: int | None, exclude_stage_groups: list[str]
) -> pd.DataFrame:
    """Patients diagnosed in or after ``min_diagnosis_year`` and not in ``exclude_stage_groups``."""
    keep = ~patients["stage_group"].isin(exclude_stage_groups)
    if min_diagnosis_year is not None:
        keep &= patients["diagnosis_year"] >= min_diagnosis_year
    return patients[keep].reset_index(drop=True)


def load_patients(patients_uri: str) -> pd.DataFrame:
    """patients.parquet written by preprocessing.clinical_dataset / mug_dataset."""
    return pd.read_parquet(download_artifacts(artifact_uri=patients_uri))


def load_splits(splits_uri: str) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """CV folds and the held-out test set written by preprocessing.dataset_split."""
    root = Path(download_artifacts(artifact_uri=splits_uri))
    fold_dirs = sorted((root / "train").glob("fold_*"), key=lambda p: int(p.name.split("_")[1]))
    if not fold_dirs:
        raise ValueError(f"No train/fold_* directories under {splits_uri}")
    folds = [pd.read_parquet(d / "patients.parquet") for d in fold_dirs]
    return folds, pd.read_parquet(root / "test" / "patients.parquet")
