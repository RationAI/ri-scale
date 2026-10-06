import logging
import tempfile
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.model_selection import StratifiedKFold, train_test_split

log = logging.getLogger(__name__)


def _strata(patients: pd.DataFrame, columns: list[str], n_folds: int) -> np.ndarray:
    """Joint strata of ``columns``; members of too small strata are stratified by the first column only.

    They join the largest stratum with the same first-column value, as a stratum
    of their own could again be too small (e.g. a single event among them).
    """
    strata = patients[columns].astype(str).agg("|".join, axis=1)
    first = patients[columns[0]].astype(str)
    small = strata.map(strata.value_counts()) < 2 * n_folds
    if small.any():
        log.info("%d patients in small %s strata, stratified by %s only", small.sum(), columns, columns[0])
        largest = strata[~small].groupby(first[~small]).agg(lambda s: s.value_counts().idxmax())
        strata = strata.where(~small, first.map(largest).fillna(first))
    return strata.to_numpy()


def create_splits(
    patients: pd.DataFrame,
    test_fraction: float,
    n_folds: int,
    stratify_by: list[str],
    seed: int,
) -> dict[str, pd.DataFrame | list[pd.DataFrame]]:
    """Hold out a test set and partition the rest into ``n_folds`` disjoint CV folds.

    Rows are patients, so no grouping is needed.

    Returns a dict with keys "test" and "train" (a list of n_folds DataFrames).
    """
    train_pool, test = train_test_split(
        patients,
        test_size=test_fraction,
        random_state=seed,
        stratify=_strata(patients, stratify_by, n_folds),
    )
    train_pool = train_pool.reset_index(drop=True)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = [
        train_pool.iloc[fold_idx].reset_index(drop=True)
        for _, fold_idx in skf.split(train_pool, _strata(train_pool, stratify_by, n_folds))
    ]
    return {"test": test.reset_index(drop=True), "train": folds}


def save_splits(splits: dict[str, pd.DataFrame | list[pd.DataFrame]], output_dir: Path) -> None:
    for name, data in splits.items():
        parts = {f"{name}/fold_{k}": df for k, df in enumerate(data)} if isinstance(data, list) else {name: data}
        for part, df in parts.items():
            (output_dir / part).mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_dir / part / "patients.parquet", index=False)


def log_split_metrics(splits: dict[str, pd.DataFrame | list[pd.DataFrame]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, data in splits.items():
        parts = {f"{name}_fold_{k}": df for k, df in enumerate(data)} if isinstance(data, list) else {name: data}
        for part, df in parts.items():
            metrics[f"{part}_n_patients"] = len(df)
            metrics[f"{part}_n_os_events"] = int(df["os_event"].sum())
            metrics[f"{part}_n_css_events"] = int(df["css_event"].sum())
    return metrics


# ── entrypoint ─────────────────────────────────────────────────────────────────

@with_cli_args(["+preprocessing=dataset_split"])
@hydra.main(
    config_path="../configs",
    config_name="preprocessing",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    patients = pd.read_parquet(download_artifacts(config.patients_uri))

    splits = create_splits(
        patients,
        test_fraction=float(config.test_fraction),
        n_folds=int(config.n_folds),
        stratify_by=list(config.stratify_by),
        seed=int(config.seed),
    )
    mlflow.log_metrics(log_split_metrics(splits))

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_splits(splits, Path(tmp_dir))
        logger.log_artifacts(tmp_dir)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
