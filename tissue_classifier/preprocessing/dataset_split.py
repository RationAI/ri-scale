from pathlib import Path

import numpy as np
import pandas as pd
from ratiopath.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold


def _stratify_labels(slides: pd.DataFrame) -> np.ndarray | None:
    y = slides["tissue_type"].values
    return y if len(np.unique(y)) > 1 else None


def create_splits(
    slides_df: pd.DataFrame,
    val_fraction: float,
    test_fraction: float,
    n_folds: int,
    seed: int,
) -> dict[str, pd.DataFrame | list[pd.DataFrame]]:
    """Split slides into val, test, and a list of n_folds train partitions.

    All slides for a patient always land in the same partition.
    Excluded slides (tissue_type == "excluded") are dropped before splitting.

    Returns a dict with keys "val", "test", and "train" (a list of n_folds DataFrames).
    """
    slides = slides_df[~slides_df["excluded"]].copy().reset_index(drop=True)

    case_ids = slides["case_id"].values
    stratify = _stratify_labels(slides)

    # First carve off the test set
    train_val, test = train_test_split(
        slides,
        test_size=test_fraction,
        random_state=seed,
        stratify=stratify,
        groups=case_ids,
    )
    train_val = train_val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    # Then carve off the val set from the remaining pool
    val_size_adjusted = val_fraction / (1.0 - test_fraction)
    stratify_tv = _stratify_labels(train_val)
    train_pool, val = train_test_split(
        train_val,
        test_size=val_size_adjusted,
        random_state=seed,
        stratify=stratify_tv,
        groups=train_val["case_id"].values,
    )
    train_pool = train_pool.reset_index(drop=True)
    val = val.reset_index(drop=True)

    # Partition train_pool into n_folds non-overlapping folds
    # Use the test indices from StratifiedGroupKFold so the folds are disjoint
    # and together cover all of train_pool exactly once.
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y = train_pool["tissue_type"].values
    groups = train_pool["case_id"].values

    train_folds: list[pd.DataFrame] = []
    for _, fold_idx in skf.split(train_pool, y=y, groups=groups):
        train_folds.append(train_pool.iloc[fold_idx].copy().reset_index(drop=True))

    return {"val": val, "test": test, "train": train_folds}


def save_splits(
    splits: dict[str, pd.DataFrame | list[pd.DataFrame]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in splits.items():
        if isinstance(data, list):
            for k, fold_df in enumerate(data):
                fold_dir = output_dir / name / f"fold_{k}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                fold_df.to_csv(fold_dir / "slides.csv", index=False)
        else:
            split_dir = output_dir / name
            split_dir.mkdir(parents=True, exist_ok=True)
            data.to_csv(split_dir / "slides.csv", index=False)


def log_split_metrics(splits: dict[str, pd.DataFrame | list[pd.DataFrame]]) -> dict[str, int]:
    metrics: dict[str, int] = {}
    for name, data in splits.items():
        if isinstance(data, list):
            for k, fold_df in enumerate(data):
                prefix = f"train_fold_{k}"
                metrics[f"{prefix}_n_slides"] = len(fold_df)
                metrics[f"{prefix}_n_cases"] = int(fold_df["case_id"].nunique())
                metrics[f"{prefix}_n_ln"] = int((fold_df["tissue_type"] == "LN").sum())
                metrics[f"{prefix}_n_colorectum"] = int((fold_df["tissue_type"] == "colorectum").sum())
        else:
            metrics[f"{name}_n_slides"] = len(data)
            metrics[f"{name}_n_cases"] = int(data["case_id"].nunique())
            metrics[f"{name}_n_ln"] = int((data["tissue_type"] == "LN").sum())
            metrics[f"{name}_n_colorectum"] = int((data["tissue_type"] == "colorectum").sum())
    return metrics
