"""Can a feature be omitted? Cross-validated drop-one-feature importance.

For every raw feature the pipeline is refitted without it on the same folds. The
delta is signed so that a positive value means the model is better *with* the
feature; deltas within the fold-to-fold noise mean the feature can be dropped.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from survival_model.data import Fold
from survival_model.evaluation import cross_validate

# metric -> sign that turns "baseline - reduced" into "positive = feature helps"
_METRICS = {"c_index": 1.0, "uno_c_index": 1.0, "ibs": -1.0}


def drop_group_importance(
    pipeline: Pipeline,
    folds: list[Fold],
    horizons: Sequence[float],
    baseline: list[dict[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (summary per feature, deltas per feature and fold)."""
    encoder = pipeline.named_steps["encoder"]
    numeric, categorical = list(encoder.numeric), list(encoder.categorical)

    rows = []
    for feature in [*numeric, *categorical]:
        if len(numeric) + len(categorical) == 1:
            break
        reduced = clone(pipeline).set_params(
            encoder__numeric=[f for f in numeric if f != feature],
            encoder__categorical=[f for f in categorical if f != feature],
        )
        fold_metrics, _ = cross_validate(reduced, folds, horizons)
        for k, (base, without) in enumerate(zip(baseline, fold_metrics, strict=True)):
            rows.append({
                "feature": feature,
                "fold": k,
                **{
                    f"delta_{m}": sign * (base[m] - without[m])
                    for m, sign in _METRICS.items()
                    if m in base and m in without
                },
            })

    per_fold = pd.DataFrame(rows)
    if per_fold.empty:
        return per_fold, per_fold
    deltas = [c for c in per_fold.columns if c.startswith("delta_")]
    summary = per_fold.groupby("feature")[deltas].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["folds_improved"] = per_fold.groupby("feature")["delta_c_index"].agg(lambda d: (d > 0).mean())
    summary = summary.sort_values("delta_c_index_mean", ascending=False).reset_index()
    return summary, per_fold


def plot_drop_group(summary: pd.DataFrame, path: Path) -> None:
    table = summary.sort_values("delta_c_index_mean")
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(table) + 1.5))
    y_pos = np.arange(len(table))
    ax.barh(y_pos, table["delta_c_index_mean"], xerr=table["delta_c_index_std"], capsize=3)
    ax.axvline(0.0, color="grey", linewidth=1)
    ax.set_yticks(y_pos, table["feature"])
    ax.set_xlabel("C-index lost when the feature is dropped (mean ± sd over folds)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
