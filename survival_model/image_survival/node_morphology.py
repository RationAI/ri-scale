"""Do basic lymph-node measurements (number, size, shape) carry survival information? Only the predicted node masks are used:
no staging, no TNM, no markers, no slide-name suffix. Age and sex are the only covariates.

extract : per patient measurements from the SAM3 node polygons of the patient's slides  -> features.csv
analyse : distribution of every measurement (normal or groups?), association with overall survival (unadjusted and adjusted for age and
          sex, Holm-corrected), tertile survival, and the gain in cross-validated C-index over age + sex  -> tests.csv, cv.csv, report.txt

The pathologist's examined-node count is read ONLY to check how well the image node count measures it; it is never a model input.

usage: python -m image_survival.node_morphology --geojson <dir> --cohort cohort_slides.json --patients-min patients_min.csv \
           --patients patients_full.parquet --out <dir> [--steps extract,analyse]
"""
import argparse
import csv
import json
import os
import warnings

import cv2
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# name -> (description, log-transform before analysis)
FEATURES = {
    "n_slides": ("slides of the patient with lymph-node tissue", True),
    "nodes_total": ("node profiles over all slides", True),
    "nodes_per_slide_median": ("median node profiles per slide", False),
    "nodes_per_slide_max": ("largest number of profiles on one slide", False),
    "area_total_mm2": ("summed area of all node profiles (mm2)", True),
    "area_per_slide_mm2": ("mean node area per slide (mm2)", True),
    "area_mean_mm2": ("mean area of a node profile (mm2)", True),
    "area_median_mm2": ("median area of a node profile (mm2)", True),
    "area_max_mm2": ("area of the largest node profile (mm2)", True),
    "area_sd_log": ("spread of node sizes (s.d. of log area)", False),
    "small_fraction": ("share of node profiles below 1 mm2", False),
    "solidity_mean": ("mean solidity: area / convex hull area, 1 = smooth round outline", False),
    "elongation_mean": ("mean short / long axis of the bounding box, 1 = round", False),
    "prob_mean": ("mean segmentation confidence", False),
}
GROUPS = {
    "count": ["n_slides", "nodes_total", "nodes_per_slide_median", "nodes_per_slide_max"],
    "size": ["area_total_mm2", "area_per_slide_mm2", "area_mean_mm2", "area_median_mm2", "area_max_mm2", "area_sd_log", "small_fraction"],
    "shape": ["solidity_mean", "elongation_mean"],
}


# ── extraction ────────────────────────────────────────────────────────────────────────────────────────

def profile_rows(geojson_path):
    """One row per predicted node profile of a slide: area (mm2), solidity, elongation, confidence."""
    with open(geojson_path) as f:
        fc = json.load(f)
    mpp = float(fc.get("mpp") or 0) or 0.2428
    rows = []
    for ft in fc["features"]:
        g = ft["geometry"]
        parts = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        pts = [np.asarray(p[0], np.float32) for p in parts if p and len(p[0]) >= 3]
        if not pts:
            continue
        pts = np.concatenate(pts)
        area_px = float(ft["properties"]["area_l0_px2"])
        hull = cv2.convexHull(pts)
        hull_area = float(cv2.contourArea(hull))
        (_, _), (w, h), _ = cv2.minAreaRect(hull)
        rows.append({"area": area_px * mpp ** 2 / 1e6, "solidity": min(1.0, area_px / hull_area) if hull_area > 0 else np.nan,
                     "elong": min(w, h) / max(w, h) if max(w, h) > 0 else np.nan, "prob": float(ft["properties"].get("mean_prob", np.nan))})
    return rows


def extract(a):
    with open(a.patients_min) as f:
        patient_of = {r["case_ids"]: r["patient_id"] for r in csv.DictReader(f)}
    with open(a.cohort) as f:
        cohort = json.load(f)
    out = []
    for case, slides in sorted(cohort.items()):
        per_slide, profiles = [], []
        for slide in sorted(slides):
            path = f"{a.geojson}/{slide}.geojson"
            if not os.path.exists(path):
                continue
            rows = profile_rows(path)
            per_slide.append((len(rows), sum(r["area"] for r in rows)))
            profiles += rows
        if not per_slide:
            continue
        area = np.array([r["area"] for r in profiles]) if profiles else np.array([np.nan])
        counts = np.array([c for c, _ in per_slide])
        out.append({
            "patient_id": patient_of[case], "n_slides": len(per_slide), "nodes_total": int(counts.sum()),
            "nodes_per_slide_median": float(np.median(counts)), "nodes_per_slide_max": int(counts.max()),
            "area_total_mm2": float(np.nansum(area)) if profiles else 0.0, "area_per_slide_mm2": float(np.mean([s for _, s in per_slide])),
            "area_mean_mm2": float(np.nanmean(area)), "area_median_mm2": float(np.nanmedian(area)), "area_max_mm2": float(np.nanmax(area)),
            "area_sd_log": float(np.nanstd(np.log(np.clip(area, 1e-3, None)))) if len(profiles) > 1 else np.nan,
            "small_fraction": float(np.mean(area < 1.0)) if profiles else np.nan,
            "solidity_mean": float(np.nanmean([r["solidity"] for r in profiles])) if profiles else np.nan,
            "elongation_mean": float(np.nanmean([r["elong"] for r in profiles])) if profiles else np.nan,
            "prob_mean": float(np.nanmean([r["prob"] for r in profiles])) if profiles else np.nan,
        })
    df = pd.DataFrame(out)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(f"{a.out}/features.csv", index=False)
    print(f"features of {len(df)} patients -> {a.out}/features.csv", flush=True)
    return df


# ── analysis ──────────────────────────────────────────────────────────────────────────────────────────

def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def distribution(v):
    """Normal curve or groups? skewness before / after log, Shapiro-Wilk of the analysed scale, 1 vs 2 Gaussian components (BIC)."""
    from scipy.stats import shapiro, skew
    from sklearn.mixture import GaussianMixture
    v = np.asarray(v, float); v = v[~np.isnan(v)]; z = ((v - v.mean()) / (v.std() or 1.0)).reshape(-1, 1)
    bic = [GaussianMixture(k, n_init=3, random_state=0).fit(z).bic(z) for k in (1, 2)]
    return {"skew": float(skew(v)), "shapiro_p": float(shapiro(v).pvalue), "dBIC_2_vs_1": float(bic[0] - bic[1])}


def analyse(a, feats=None):
    from lifelines import CoxPHFitter
    from lifelines.statistics import multivariate_logrank_test
    from scipy.stats import chi2, spearmanr
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    from sksurv.util import Surv

    feats = feats if feats is not None else pd.read_csv(f"{a.out}/features.csv")
    p = pd.read_parquet(a.patients) if a.patients.endswith(".parquet") else pd.read_csv(a.patients)
    df = feats.merge(p[["patient_id", "age", "sex", "time_days", "os_event"] + (["nodes_examined"] if "nodes_examined" in p else [])], on="patient_id")
    df = df.dropna(subset=["age", "time_days"]).reset_index(drop=True)
    df["T"], df["E"], df["male"] = df["time_days"] / 365.25, df["os_event"].astype(int), (df["sex"] == "M").astype(float)
    df["agez"] = (df["age"] - df["age"].mean()) / df["age"].std()
    names = list(FEATURES)
    X = pd.DataFrame({n: (np.log1p(df[n]) if FEATURES[n][1] else df[n]) for n in names})
    X = X.fillna(X.median())
    constant = [c for c in names if not X[c].std() > 0]   # a measurement that never varies cannot be tested
    names = [c for c in names if c not in constant]
    X = X[names]
    Z = (X - X.mean()) / X.std()
    n, d = len(df), int(df["E"].sum())
    lines = [f"patients {n} | deaths {d} | median follow-up of survivors {df.loc[df.E == 0, 'T'].median():.1f} years"
             + (f" | constant, not tested: {', '.join(constant)}" if constant else ""),
             f"smallest detectable hazard ratio per s.d. (80% power): {np.exp((1.96 + 0.84) / np.sqrt(d)):.2f} for one test, "
             f"{np.exp((2.9 + 0.84) / np.sqrt(d)):.2f} after correcting for {len(names)} tests", ""]

    base = df[["T", "E", "agez", "male"]]
    ll0 = CoxPHFitter().fit(base, "T", "E").log_likelihood_
    rows = []
    for name in names:
        z = Z[name]; row = {"feature": name, **distribution(X[name]), "spearman_age": float(spearmanr(z, df["age"])[0]),
                            "median": float(df[name].median()), "iqr_lo": float(df[name].quantile(.25)), "iqr_hi": float(df[name].quantile(.75))}
        m = CoxPHFitter().fit(pd.DataFrame({"T": df["T"], "E": df["E"], "z": z}), "T", "E")
        row.update(hr_uni=float(np.exp(m.params_["z"])), p_uni=float(m.summary.loc["z", "p"]))
        m = CoxPHFitter().fit(pd.concat([base, z.rename("z")], axis=1), "T", "E")
        ci = np.exp(m.confidence_intervals_.loc["z"].to_numpy())
        row.update(hr_adj=float(np.exp(m.params_["z"])), hr_lo=float(ci[0]), hr_hi=float(ci[1]), p_adj=float(chi2.sf(2 * (m.log_likelihood_ - ll0), 1)))
        tert = pd.qcut(z.rank(method="first"), 3, labels=[0, 1, 2]).astype(int)
        row["p_logrank_tertiles"] = float(multivariate_logrank_test(df["T"], tert, df["E"]).p_value)
        dm = pd.get_dummies(tert, prefix="t", drop_first=True).astype(float)
        mt = CoxPHFitter().fit(pd.concat([base, dm], axis=1), "T", "E")
        row.update(hr_t2=float(np.exp(mt.params_["t_1"])), hr_t3=float(np.exp(mt.params_["t_2"])), p_tertiles=float(chi2.sf(2 * (mt.log_likelihood_ - ll0), 2)))
        rows.append(row)
    tests = pd.DataFrame(rows)
    tests["p_adj_holm"] = holm(tests["p_adj"])
    tests["p_uni_holm"] = holm(tests["p_uni"])

    # repeated cross-validation on shared folds: gain of C-index over age + sex
    y = Surv.from_arrays(df["E"].astype(bool).to_numpy(), df["T"].to_numpy())
    clin = np.column_stack([df["agez"], df["male"]])
    sets = {"age+sex": [], **{n: [n] for n in names}, **{f"all {g}": [c for c in cols if c in names] for g, cols in GROUPS.items()}, "all measurements": names}
    cv = []
    for f, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=a.repeats, random_state=42).split(np.zeros(n), df["E"])):
        for sname, cols in sets.items():
            M = np.column_stack([clin] + ([Z[cols].to_numpy()] if cols else []))
            mu, sd = M[tr].mean(0), M[tr].std(0); sd[sd == 0] = 1
            model = CoxPHSurvivalAnalysis(alpha=1.0 if len(cols) > 3 else 0.1).fit((M[tr] - mu) / sd, y[tr])
            cv.append({"set": sname, "repeat": f // 5, "fold": f % 5, "c": concordance_index_censored(y[te]["event"], y[te]["time"], model.predict((M[te] - mu) / sd))[0]})
    cv = pd.DataFrame(cv); base_c = cv[cv.set == "age+sex"].set_index(["repeat", "fold"]).c; cvrows = []
    for sname, g in cv.groupby("set"):
        g = g.set_index(["repeat", "fold"]); dlt = g.c - base_c; rep = dlt.groupby("repeat").mean()
        cvrows.append({"set": sname, "c_index": g.c.mean(), "gain": dlt.mean(), "sd_repeat_means": rep.std(), "repeats_positive": f"{(rep > 0).sum()}/{len(rep)}", "folds_better": (dlt > 0).mean()})
    cvt = pd.DataFrame(cvrows).sort_values("gain", ascending=False)

    # measurement check (not a model input): image node count vs the pathologist's examined nodes
    check = ""
    if "nodes_examined" in df and df["nodes_examined"].notna().sum() > 20:
        ok = df["nodes_examined"].notna()
        check = "measurement check, Spearman with the pathologist's examined-node count: " + ", ".join(
            f"{k} {spearmanr(df.loc[ok, k], df.loc[ok, 'nodes_examined'])[0]:+.2f}" for k in ("n_slides", "nodes_total", "nodes_per_slide_median", "nodes_per_slide_max", "area_total_mm2"))

    cor = Z.corr().abs().where(np.triu(np.ones((len(names), len(names))), 1).astype(bool)).stack().sort_values(ascending=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    show = tests[["feature", "median", "iqr_lo", "iqr_hi", "skew", "shapiro_p", "dBIC_2_vs_1", "spearman_age", "hr_adj", "hr_lo", "hr_hi", "p_adj", "p_adj_holm", "p_logrank_tertiles", "hr_t2", "hr_t3"]]
    lines += ["DISTRIBUTION AND ASSOCIATION (analysed scale = log for count and area measures; HR per 1 s.d., adjusted for age and sex)",
              show.round(3).to_string(index=False), "", "CROSS-VALIDATED C-INDEX GAIN OVER AGE + SEX (5 folds x %d repeats, shared folds)" % a.repeats,
              cvt.round(3).to_string(index=False), "", check, "", "most correlated measurement pairs (|r|):"] + [f"  {i} ~ {j}: {v:.2f}" for (i, j), v in cor.head(6).items()]
    os.makedirs(a.out, exist_ok=True)
    tests.to_csv(f"{a.out}/tests.csv", index=False); cvt.to_csv(f"{a.out}/cv.csv", index=False)
    open(f"{a.out}/report.txt", "w").write("\n".join(lines))
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geojson", default=""); ap.add_argument("--cohort", default=""); ap.add_argument("--patients-min", default="")
    ap.add_argument("--patients", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", default="extract,analyse"); ap.add_argument("--repeats", type=int, default=10)
    a = ap.parse_args(); steps = a.steps.split(",")
    feats = extract(a) if "extract" in steps else None
    if "analyse" in steps:
        analyse(a, feats)


if __name__ == "__main__":
    main()
