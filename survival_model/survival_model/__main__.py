import tempfile
from pathlib import Path

import hydra
import joblib
import mlflow
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from survival_model import significance
from survival_model.data import Fold, load_splits
from survival_model.evaluation import (
    bootstrap_c_index,
    coefficients,
    cross_validate,
    evaluate,
    plot_risk_groups,
    predict_frame,
    summarize,
)
from survival_model.features import TabularEncoder
from survival_model.importance import drop_group_importance, plot_drop_group


@hydra.main(config_path="../configs", config_name="base", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    np.random.seed(config.seed)
    horizons = list(config.horizons)

    train_dfs, test_df = load_splits(config.data.splits_uri)
    folds = [Fold.from_patients(df, config.endpoint) for df in train_dfs]
    train = Fold.concat(folds)

    features = OmegaConf.to_container(config.features, resolve=True)
    pipeline = Pipeline([
        ("encoder", TabularEncoder(**features)),  # type: ignore[arg-type]
        ("model", instantiate(config.model)),
    ])

    fold_metrics, cv_predictions = cross_validate(pipeline, folds, horizons)
    mlflow.log_metrics(summarize(fold_metrics, "cv"))

    final = clone(pipeline).fit(train.X, train.y)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        pd.DataFrame(fold_metrics).rename_axis("fold").to_csv(tmp / "cv_metrics.csv")
        cv_predictions.to_csv(tmp / "cv_predictions.csv", index=False)
        plot_risk_groups(cv_predictions, tmp / "cv_risk_groups.png")
        if (coefs := coefficients(final)) is not None:
            coefs.to_csv(tmp / "coefficients.csv", index=False)

        # One table answering "which features matter / can be omitted"
        summary = pd.DataFrame({"feature": [*pipeline["encoder"].numeric, *pipeline["encoder"].categorical]})
        if config.analysis.significance:
            tests, metrics = significance.analyse(
                train.X, train.y, final["encoder"], float(config.analysis.penalizer), tmp
            )
            mlflow.log_metrics(metrics)
            summary = summary.merge(tests, on="feature", how="left")
        if config.analysis.drop_group:
            importance, per_fold = drop_group_importance(pipeline, folds, horizons, fold_metrics)
            if not importance.empty:
                per_fold.to_csv(tmp / "drop_group_per_fold.csv", index=False)
                plot_drop_group(importance, tmp / "drop_group_importance.png")
                summary = summary.merge(importance, on="feature", how="left")
        summary.to_csv(tmp / "feature_summary.csv", index=False)

        if config.evaluate_test:
            test = Fold.from_patients(test_df, config.endpoint)
            test_metrics = evaluate(final, test.X, test.y, train.y, horizons)
            test_metrics |= bootstrap_c_index(final.predict(test.X), test.y, seed=config.seed)
            mlflow.log_metrics({f"test/{k}": v for k, v in test_metrics.items()})
            predict_frame(final, test, horizons).to_csv(tmp / "test_predictions.csv", index=False)

        # Fitted on the whole training pool; joblib.load(...).predict_survival_function(derive_features(df))
        joblib.dump(final, tmp / "model.joblib")
        logger.log_artifacts(tmp_dir)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
