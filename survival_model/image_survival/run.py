"""Repeated patient-level cross-validation of the image models against the clinical baseline.

With a few hundred patients a single split says little, so every model is scored on the same
``--repeats`` x ``--folds`` stratified folds and compared fold by fold. Hyper-parameters are fixed, not tuned on the folds.

usage: python -m image_survival.run --nodes nodes.npz --patients patients_min.csv --out results/
"""
import argparse, json, os, time

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc

from image_survival.data import Cohort
from image_survival.models import AttentionMIL, ClinicalCox, PooledCox

MODELS = {
    "clinical": lambda: ClinicalCox(),
    "pooled_pca8": lambda: PooledCox(n_components=8),
    "pooled_pca8_hand": lambda: PooledCox(n_components=8, hand=True),
    "pooled_pca16": lambda: PooledCox(n_components=16),
    "mil": lambda: AttentionMIL(),
    "mil_image_only": lambda: AttentionMIL(use_clinical=False),
}


def score(y_train, y_test, risk, horizons=(3.0, 5.0)) -> dict:
    out = {"c_index": concordance_index_censored(y_test["event"], y_test["time"], risk)[0]}
    inside = y_test["time"] < y_train["time"].max()
    try:
        out["uno_c_index"] = concordance_index_ipcw(y_train, y_test[inside], risk[inside], tau=min(max(horizons), y_train["time"].max() * 0.99))[0]
        for h in horizons:
            if y_test["time"][inside].min() < h < y_test["time"][inside].max():
                out[f"auc_{h:g}y"] = cumulative_dynamic_auc(y_train, y_test[inside], risk[inside], [h])[0][0]
    except ValueError:
        pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True); ap.add_argument("--patients", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=",".join(MODELS)); ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--margin", action="store_true", help="use the margin embeddings instead of the node core")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cohort = Cohort(a.nodes, a.patients, margin=a.margin)
    print(json.dumps(cohort.summary()), flush=True)
    names = a.models.split(",")
    splitter = RepeatedStratifiedKFold(n_splits=a.folds, n_repeats=a.repeats, random_state=a.seed)
    rows = []
    for f, (train, test) in enumerate(splitter.split(np.zeros(len(cohort)), cohort.y["event"])):
        for name in names:
            t0 = time.time()
            risk = MODELS[name]().fit(cohort, train).risk(cohort, test)
            rows.append({"model": name, "repeat": f // a.folds, "fold": f % a.folds, **score(cohort.y[train], cohort.y[test], risk)})
        print(f"fold {f + 1}/{a.folds * a.repeats} done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{a.out}/fold_metrics.csv", index=False)

    metrics = [c for c in df.columns if c not in ("model", "repeat", "fold")]
    summary = df.groupby("model")[metrics].agg(["mean", "std"]).reindex(names)
    base = df[df.model == "clinical"].set_index(["repeat", "fold"])["c_index"] if "clinical" in names else None
    if base is not None:
        gain = {n: (df[df.model == n].set_index(["repeat", "fold"])["c_index"] - base) for n in names if n != "clinical"}
        summary[("c_index_gain_vs_clinical", "mean")] = pd.Series({n: g.mean() for n, g in gain.items()})
        summary[("folds_better_than_clinical", "frac")] = pd.Series({n: (g > 0).mean() for n, g in gain.items()})
    summary.round(3).to_csv(f"{a.out}/summary.csv")
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
