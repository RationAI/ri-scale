import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)

from survival_model.data import Fold

log = logging.getLogger(__name__)


def survival_at(model: Pipeline, X: pd.DataFrame, times: Sequence[float]) -> np.ndarray:
    """Predicted S(t) for every row of X, shape (n_samples, n_times)."""
    t = np.asarray(times, dtype=float)
    return np.vstack([fn(np.clip(t, *fn.domain)) for fn in model.predict_survival_function(X)])


def evaluate(
    model: Pipeline, X: pd.DataFrame, y: np.ndarray, y_train: np.ndarray, horizons: Sequence[float]
) -> dict[str, float]:
    """Harrell's and Uno's C-index, time-dependent AUC and Brier score at the horizons, IBS."""
    risk = model.predict(X)
    metrics = {"c_index": concordance_index_censored(y["event"], y["time"], risk)[0]}

    # The IPCW metrics estimate the censoring distribution on the training data,
    # so only test times inside the training follow-up can be weighted.
    keep = y["time"] < y_train["time"].max()
    X, y, risk = X[keep], y[keep], risk[keep]
    t_min, t_max = y["time"].min(), y["time"].max()
    times = [h for h in horizons if t_min <= h < t_max]
    try:
        metrics["uno_c_index"] = concordance_index_ipcw(y_train, y, risk, tau=min(max(horizons), t_max))[0]
        if times:
            auc, _ = cumulative_dynamic_auc(y_train, y, risk, times)
            _, brier = brier_score(y_train, y, survival_at(model, X, times), times)
            for h, auc_h, brier_h in zip(times, auc, brier, strict=True):
                metrics[f"auc_{h}y"] = auc_h
                metrics[f"brier_{h}y"] = brier_h
        grid = np.linspace(*np.percentile(y["time"], [10, 90]), num=50)
        metrics["ibs"] = integrated_brier_score(y_train, y, survival_at(model, X, grid), grid)
    except ValueError as e:
        log.warning("Skipping IPCW metrics: %s", e)

    return {k: float(v) for k, v in metrics.items()}


def bootstrap_c_index(
    risk: np.ndarray, y: np.ndarray, n_resamples: int = 2000, seed: int = 0
) -> dict[str, float]:
    """95 % percentile bootstrap interval of Harrell's C-index, resampling patients."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(y), len(y))
        try:
            values.append(concordance_index_censored(y["event"][idx], y["time"][idx], risk[idx])[0])
        except ValueError:  # no comparable pairs in the resample
            continue
    lower, upper = np.percentile(values, [2.5, 97.5])
    return {"c_index_ci_lower": float(lower), "c_index_ci_upper": float(upper)}


def predict_frame(model: Pipeline, data: Fold, horizons: Sequence[float]) -> pd.DataFrame:
    survival = survival_at(model, data.X, horizons)
    return pd.DataFrame({
        "patient_id": data.patient_id.to_numpy(),
        "time": data.y["time"],
        "event": data.y["event"],
        "risk": model.predict(data.X),
        **{f"survival_{h}y": survival[:, i] for i, h in enumerate(horizons)},
    })


def cross_validate(
    pipeline: Pipeline, folds: list[Fold], horizons: Sequence[float]
) -> tuple[list[dict[str, float]], pd.DataFrame]:
    """Fit on all folds but one, evaluate on the held-out one; returns per-fold metrics and OOF predictions."""
    fold_metrics, predictions = [], []
    for k, held_out in enumerate(folds):
        train = Fold.concat([f for i, f in enumerate(folds) if i != k])
        model = clone(pipeline).fit(train.X, train.y)
        fold_metrics.append(evaluate(model, held_out.X, held_out.y, train.y, horizons))
        predictions.append(predict_frame(model, held_out, horizons).assign(fold=k))
    return fold_metrics, pd.concat(predictions, ignore_index=True)


def summarize(fold_metrics: list[dict[str, float]], prefix: str) -> dict[str, float]:
    df = pd.DataFrame(fold_metrics)
    summary: dict[str, float] = {}
    for name in df.columns:
        summary[f"{prefix}/{name}/mean"] = df[name].mean()
        summary[f"{prefix}/{name}/std"] = df[name].std()
        for k, value in df[name].items():
            summary[f"{prefix}/fold_{k}/{name}"] = value
    return {k: float(v) for k, v in summary.items() if not np.isnan(v)}


def coefficients(model: Pipeline) -> pd.DataFrame | None:
    """Coefficients of a linear survival model (hazard ratios per encoded column; per SD if scaled)."""
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "coef_"):
        return None
    coef = np.asarray(estimator.coef_)
    coef = coef[:, -1] if coef.ndim == 2 else coef  # Coxnet: last alpha of the path
    return pd.DataFrame({
        "column": model.named_steps["encoder"].get_feature_names_out(),
        "coef": coef,
        "hazard_ratio": np.exp(coef),
    })


def plot_risk_groups(predictions: pd.DataFrame, path: Path) -> None:
    """Kaplan-Meier curves of out-of-fold risk tertiles (tertiles computed within each fold)."""
    labels = ["low", "intermediate", "high"]
    groups = predictions.groupby("fold")["risk"].transform(
        lambda r: pd.qcut(r.rank(method="first"), 3, labels=labels).astype(str)
    )
    test = multivariate_logrank_test(predictions["time"], groups, predictions["event"])

    fig, ax = plt.subplots(figsize=(6, 4))
    for label in labels:
        mask = groups == label
        KaplanMeierFitter().fit(
            predictions.loc[mask, "time"], predictions.loc[mask, "event"], label=f"{label} (n={mask.sum()})"
        ).plot_survival_function(ax=ax)
    ax.set_xlabel("Years since diagnosis")
    ax.set_ylabel("Survival probability")
    ax.set_title(f"Out-of-fold risk tertiles (log-rank p={test.p_value:.2g})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
