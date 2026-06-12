import tempfile
from pathlib import Path

import hydra
import mlflow
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from preprocessing.dataset_builder import build_patients_df, build_slides_df


@with_cli_args(["+preprocessing=dataset_build"])
@hydra.main(
    config_path="../configs",
    config_name="preprocessing",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides_dir = Path(config.slides_dir)
    slides_df = build_slides_df(slides_dir, max_workers=config.max_workers)
    patients_df = build_patients_df(slides_df)

    active = slides_df[~slides_df["excluded"]]
    mlflow.log_metrics({
        "n_slides_total": len(slides_df),
        "n_slides_active": len(active),
        "n_slides_excluded": int(slides_df["excluded"].sum()),
        "n_slides_ln": int((active["tissue_type"] == "LN").sum()),
        "n_slides_colorectum": int((active["tissue_type"] == "colorectum").sum()),
        "n_cases_total": int(active["case_id"].nunique()),
    })

    with tempfile.TemporaryDirectory() as tmp_dir:
        slides_df.to_csv(f"{tmp_dir}/slides.csv", index=False)
        patients_df.to_csv(f"{tmp_dir}/patients.csv", index=False)
        logger.log_artifacts(tmp_dir)

    mlflow.log_input(
        mlflow.data.from_pandas(slides_df, name="slides"),
        context="slides",
    )
    mlflow.log_input(
        mlflow.data.from_pandas(patients_df, name="patients"),
        context="patients",
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
