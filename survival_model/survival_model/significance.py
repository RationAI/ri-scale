"""Which features carry prognostic information? Classical Cox statistics on the training pool.

* univariable: each feature alone (likelihood-ratio test against the null model)
* adjusted: likelihood-ratio test for dropping the feature from the full model,
  all of its dummy columns at once, Holm-corrected across features
* proportional-hazards assumption test of the full model
"""

import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.exceptions import ConvergenceError
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
from scipy.stats import chi2
from sklearn.base import clone

from survival_model.features import UNKNOWN, TabularEncoder

log = logging.getLogger(__name__)


def _fit_cox(data: pd.DataFrame, penalizer: float) -> CoxPHFitter:
    try:
        return CoxPHFitter(penalizer=penalizer).fit(data, duration_col="time", event_col="event")
    except ConvergenceError:
        if penalizer >= 0.01:
            raise
        log.warning("Cox model did not converge, refitting with penalizer=0.01")
        return CoxPHFitter(penalizer=0.01).fit(data, duration_col="time", event_col="event")


def holm(p: pd.Series) -> pd.Series:
    """Holm-Bonferroni adjusted p-values."""
    ordered = p.dropna().sort_values()
    m = len(ordered)
    adjusted = (ordered * (m - np.arange(m))).cummax().clip(upper=1.0)
    return adjusted.reindex(p.index)


def _hazard_ratios(cph: CoxPHFitter) -> pd.DataFrame:
    s = cph.summary
    return pd.DataFrame({
        "hazard_ratio": s["exp(coef)"],
        "ci_lower": s["exp(coef) lower 95%"],
        "ci_upper": s["exp(coef) upper 95%"],
        "p": s["p"],
    }).rename_axis("column").reset_index()


def _design(
    X: pd.DataFrame, y: np.ndarray, encoder: TabularEncoder, features: list[str]
) -> tuple[pd.DataFrame, TabularEncoder]:
    """Cox design matrix of ``features`` with an encoder fitted on just them.

    Unscaled, so hazard ratios are per natural unit (age: per year).
    """
    encoder = clone(encoder).set_params(
        numeric=[f for f in encoder.numeric if f in features],
        categorical=[f for f in encoder.categorical if f in features],
        scale_numeric=False,
    ).fit(X)
    return encoder.transform(X).assign(time=y["time"], event=y["event"].astype(int)), encoder


def analyse(
    X: pd.DataFrame,
    y: np.ndarray,
    encoder: TabularEncoder,
    penalizer: float,
    out_dir: Path,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Writes the tables and plots into ``out_dir``; returns the per-feature test table and metrics."""
    data, full_encoder = _design(X, y, encoder, [*encoder.numeric, *encoder.categorical])
    groups = {feature: columns for feature, columns in full_encoder.feature_groups_.items() if columns}
    feature_of = {column: feature for feature, columns in groups.items() for column in columns}
    full = _fit_cox(data, penalizer)

    univariable, tests = [], []
    for feature in groups:
        single, _ = _design(X, y, encoder, [feature])
        cph = _fit_cox(single, penalizer)
        test = cph.log_likelihood_ratio_test()
        univariable.append(_hazard_ratios(cph).assign(feature=feature))

        if len(groups) == 1:
            lr, df = full.log_likelihood_ratio_test().test_statistic, len(groups[feature])
        else:
            reduced, _ = _design(X, y, encoder, [f for f in groups if f != feature])
            lr = 2 * (full.log_likelihood_ - _fit_cox(reduced, penalizer).log_likelihood_)
            df = len(data.columns) - len(reduced.columns)  # columns deduplicated away may come back
        tests.append({
            "feature": feature,
            "univariable_df": len(single.columns) - 2,
            "univariable_chi2": test.test_statistic,
            "univariable_p": test.p_value,
            "adjusted_df": df,
            "adjusted_chi2": lr,
            "adjusted_p": chi2.sf(lr, df) if df > 0 else float("nan"),
        })
    tests_df = pd.DataFrame(tests)
    tests_df["adjusted_p_holm"] = holm(tests_df["adjusted_p"])
    tests_df = tests_df.sort_values("adjusted_p").reset_index(drop=True)

    multivariable = _hazard_ratios(full).assign(feature=lambda d: d["column"].map(feature_of))
    ph_test = proportional_hazard_test(full, data, time_transform="rank").summary
    ph_test = ph_test.rename_axis("column").reset_index()

    pd.concat(univariable, ignore_index=True).to_csv(out_dir / "cox_univariable.csv", index=False)
    multivariable.to_csv(out_dir / "cox_multivariable.csv", index=False)
    tests_df.to_csv(out_dir / "feature_tests.csv", index=False)
    ph_test.to_csv(out_dir / "proportional_hazards_test.csv", index=False)
    _forest_plot(multivariable, out_dir / "forest_plot.png")
    _km_grid(X, y, [f for f in encoder.categorical if f in groups], out_dir / "km_curves.png")

    return tests_df, {
        "significance/train_c_index": float(full.concordance_index_),
        "significance/penalizer": float(full.penalizer),
        "significance/n_ph_violations": int((ph_test["p"] < 0.05).sum()),
    }


def _forest_plot(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(table) + 1.5))
    y_pos = np.arange(len(table))[::-1]
    ax.errorbar(
        table["hazard_ratio"],
        y_pos,
        xerr=[table["hazard_ratio"] - table["ci_lower"], table["ci_upper"] - table["hazard_ratio"]],
        fmt="o",
        capsize=3,
    )
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(y_pos, table["column"])
    ax.set_xlabel("Hazard ratio (95% CI), multivariable Cox")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _km_grid(X: pd.DataFrame, y: np.ndarray, features: list[str], path: Path) -> None:
    if not features:
        return
    n_cols = min(4, len(features))
    n_rows = math.ceil(len(features) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows), squeeze=False)
    for ax, feature in zip(axes.flat, features, strict=False):
        levels = X[feature].astype(object).where(X[feature].notna(), UNKNOWN).astype(str)
        test = multivariate_logrank_test(y["time"], levels, y["event"])
        for level in sorted(levels.unique()):
            mask = (levels == level).to_numpy()
            KaplanMeierFitter().fit(
                y["time"][mask], y["event"][mask], label=f"{level} ({mask.sum()})"
            ).plot_survival_function(ax=ax, ci_show=False)
        ax.set_title(f"{feature}  (log-rank p={test.p_value:.2g})", fontsize=9)
        ax.set_xlabel("Years")
        ax.legend(fontsize=7)
    for ax in axes.flat[len(features):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
