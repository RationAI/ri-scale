from pathlib import Path

import hydra
import mlflow
import pandas as pd
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from preprocessing.dataset_split import create_splits, log_split_metrics, save_splits


@with_cli_args(["+preprocessing=dataset_split"])
@hydra.main(
    config_path="../configs",
    config_name="preprocessing",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides_df = pd.read_csv(download_artifacts(config.slides_uri))
    output_dir = Path(config.output_dir)

    splits = create_splits(
        slides_df=slides_df,
        val_fraction=float(config.val_fraction),
        test_fraction=float(config.test_fraction),
        n_folds=int(config.n_folds),
        seed=int(config.seed),
    )

    save_splits(splits, output_dir)

    metrics = log_split_metrics(splits)
    mlflow.log_metrics(metrics)

    logger.log_artifacts(str(output_dir))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
