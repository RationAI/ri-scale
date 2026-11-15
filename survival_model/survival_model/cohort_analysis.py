"""Describe the tabular cohorts (MMCI, MUG): single variables AND combinations of variables.

A variable that looks balanced on its own can hide a skew in combination (all the young patients one sex, one decade full of stage IV,
a marker recorded only for the old), and such a skew confounds every survival result. For every cohort this reports

  1. overview: patients, events, follow-up, missingness, age distribution (normal or peaks?)
  2. ALL pairs of variables: Cramer's V (categorical x categorical), correlation ratio eta (numeric x categorical), Spearman rho
     (numeric x numeric), BH-corrected p-values; pairs that are true by definition (pN and the node counts, stage and TNM) are marked
  3. chosen combinations as cross-tables with standardised residuals: age band x sex, era x sex, age x stage, sex x stage, ...
  4. who is where, adjusted: logistic regression of stage, grade, nodes, site, marker on age, sex and era together
  5. survival in combinations: 5-year survival per age band x sex, interaction tests (Cox likelihood-ratio), univariable against
     adjusted hazard ratios (confounding)
  6. missingness: is a variable missing more often in the old, the male, an era, the dead?
  7. the young (< 50) against the rest
  8. between cohorts: standardised differences of the shared variables and how well a model tells the cohorts apart (domain AUC)

usage: python -m survival_model.cohort_analysis --cohort MMCI=<patients.parquet or mlflow uri> --cohort MUG=<...> --out <dir>
"""
import argparse
import itertools
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from survival_model.features import derive_features

warnings.filterwarnings("ignore")

CATEGORICAL = ["sex", "age_band", "era", "location", "morphology_group", "grade_group", "pt_group", "pn_group", "m_group", "stage_group",
               "neoadjuvant", "mmr", "kras", "nras", "braf"]
NUMERIC = ["age", "diagnosis_year", "nodes_examined", "nodes_positive", "ln_ratio"]
# true by definition, not findings: pN and stage come from the TNM / node counts
STRUCTURAL = [{"pn_group", "nodes_positive"}, {"pn_group", "ln_ratio"}, {"nodes_positive", "ln_ratio"}, {"nodes_examined", "ln_ratio"},
              {"stage_group", "pt_group"}, {"stage_group", "pn_group"}, {"stage_group", "m_group"}, {"stage_group", "nodes_positive"},
              {"stage_group", "ln_ratio"}, {"age", "age_band"}, {"diagnosis_year", "era"}]
REFERENCE = {"stage_group": "0-I", "pt_group": "T0-2", "pn_group": "N0", "m_group": "M0", "grade_group": "low", "sex": "F",
             "location": "left", "mmr": "pMMR", "kras": "wt", "nras": "wt", "braf": "wt", "neoadjuvant": "no", "morphology_group": "adenocarcinoma", "age_band": "60-69"}
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
MIN_LEVEL = 10   # levels with fewer patients are merged into 'other (rare)' (variables with two levels are never merged)


# ── data ──────────────────────────────────────────────────────────────────────────────────────────────

def load(uri: str) -> pd.DataFrame:
    if os.path.exists(uri):
        return pd.read_parquet(uri)
    from survival_model.data import load_patients
    return load_patients(uri)


def prepare(patients: pd.DataFrame) -> pd.DataFrame:
    f = derive_features(patients)
    f["age_band"] = pd.cut(f["age"], [0, 50, 60, 70, 80, 200], right=False, labels=["<50", "50-59", "60-69", "70-79", "80+"]).astype(object)
    year = f["diagnosis_year"]
    f["era"] = pd.cut(year, [0, 1990, 2000, 2010, 2100], right=False, labels=["<1990", "1990s", "2000s", "2010+"]).astype(object)
    f["os_years"] = patients["time_days"].astype(float) / 365.25
    f["event"] = patients["os_event"].astype(bool)
    f["male"] = (f["sex"] == "M").astype(float).where(f["sex"].notna())
    f["pn_positive"] = f["pn_group"].map(lambda v: np.nan if pd.isna(v) else float(v != "N0"))
    f["stage_late"] = f["stage_group"].map(lambda v: np.nan if pd.isna(v) else float(v in ("III", "IV")))
    f["grade_high"] = f["grade_group"].map({"low": 0.0, "high": 1.0})
    f["pt4"] = f["pt_group"].map(lambda v: np.nan if pd.isna(v) else float(v == "T4"))
    f["right_sided"] = f["location"].map(lambda v: np.nan if pd.isna(v) else float(v == "right"))
    f["dmmr"] = f["mmr"].map(lambda v: np.nan if pd.isna(v) else float(v == "dMMR"))
    f["young"] = (f["age"] < 50).astype(float).where(f["age"].notna())
    for col in CATEGORICAL:                                       # rare levels do not carry a cross-table
        if col in f:
            counts = f[col].value_counts()
            if len(counts) > 2:
                f[col] = f[col].where(~f[col].isin(counts[counts < MIN_LEVEL].index) | f[col].isna(), "other (rare)")
    keep_cat = [c for c in CATEGORICAL if c in f and f[c].nunique() >= 2]
    keep_num = [c for c in NUMERIC if c in f and f[c].nunique() >= 3]
    f.attrs["cat"], f.attrs["num"] = keep_cat, keep_num
    return f


def bh(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p); q = np.empty(n); run = 1.0
    for rank, i in zip(range(n, 0, -1), order[::-1]):
        run = min(run, p[i] * n / rank); q[i] = run
    return q


def hist(x, bins=36):
    cnt, edges = np.histogram(x, bins=bins)
    blocks = " ▁▂▃▄▅▆▇█"
    return f"{edges[0]:.0f} " + "".join(blocks[int(round(c / cnt.max() * 8))] for c in cnt) + f" {edges[-1]:.0f}"


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


# ── association measures ──────────────────────────────────────────────────────────────────────────────

def cramers_v(a, b):
    t = pd.crosstab(a, b)
    if t.shape[0] < 2 or t.shape[1] < 2:
        return np.nan, np.nan, len(a)
    chi2, p, _, _ = stats.chi2_contingency(t, correction=False)
    n = t.values.sum(); r, k = t.shape
    phi2c = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1)); rc = r - (r - 1) ** 2 / (n - 1); kc = k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2c / max(1e-9, min(kc - 1, rc - 1)))), float(p), int(n)


def eta(x, g):
    groups = [v.values for _, v in x.groupby(g) if len(v) >= 2]
    if len(groups) < 2:
        return np.nan, np.nan, len(x)
    allv = np.concatenate(groups); ssb = sum(len(v) * (v.mean() - allv.mean()) ** 2 for v in groups); sst = ((allv - allv.mean()) ** 2).sum()
    return float(np.sqrt(ssb / sst)) if sst > 0 else np.nan, float(stats.kruskal(*groups).pvalue), len(allv)


def association(f, a, b):
    d = f[[a, b]].dropna()
    if len(d) < 30:
        return np.nan, np.nan, len(d), "-"
    num_a, num_b = a in f.attrs["num"], b in f.attrs["num"]
    if num_a and num_b:
        r, p = stats.spearmanr(d[a], d[b]); return float(abs(r)), float(p), len(d), "|rho|"
    if num_a or num_b:
        x, g = (a, b) if num_a else (b, a)
        v, p, n = eta(d[x].astype(float), d[g]); return v, p, n, "eta"
    v, p, n = cramers_v(d[a], d[b]); return v, p, n, "V"


def all_pairs(f):
    cols = f.attrs["cat"] + f.attrs["num"]
    rows = []
    for a, b in itertools.combinations(cols, 2):
        v, p, n, kind = association(f, a, b)
        rows.append({"a": a, "b": b, "effect": v, "p": p, "n": n, "measure": kind, "structural": any({a, b} == s for s in STRUCTURAL)})
    df = pd.DataFrame(rows).dropna(subset=["p"])
    df["q_BH"] = bh(df["p"].values)
    return df.sort_values("effect", ascending=False).reset_index(drop=True)


# ── cross-tables ──────────────────────────────────────────────────────────────────────────────────────

def cross_table(f, a, b, lines, always=False, min_v=0.1):
    d = f[[a, b]].dropna()
    t = pd.crosstab(d[a], d[b])
    if t.shape[0] < 2 or t.shape[1] < 2 or len(d) < 30:
        return
    v, p, n = cramers_v(d[a], d[b])
    chi2, _, _, exp = stats.chi2_contingency(t, correction=False)
    resid = (t.values - exp) / np.sqrt(exp * (1 - t.sum(1).values[:, None] / n) * (1 - t.sum(0).values[None, :] / n))   # adjusted residuals
    flagged = [(t.index[i], t.columns[j], int(t.iat[i, j]), exp[i, j], resid[i, j]) for i in range(t.shape[0]) for j in range(t.shape[1]) if abs(resid[i, j]) >= 3]
    if not always and (v < min_v and not flagged):
        lines.append(f"  {a} x {b}: n={n}, V={v:.2f}, p={p:.2g}: no notable combination")
        return
    share = (t.div(t.sum(1), axis=0) * 100).round(0).astype(int)
    lines += [f"  {a} (rows) x {b} (columns): n={n}, Cramer's V={v:.2f}, chi2 p={p:.2g}",
              "    counts (row %)   " + "   ".join(f"{c:>12s}" for c in t.columns)]
    lines += [f"    {str(r):16s} " + "   ".join(f"{int(t.at[r, c]):5d} ({share.at[r, c]:3d}%)" for c in t.columns) for r in t.index]
    if flagged:
        lines.append("    cells far from independence (|adjusted residual| >= 3): " + "; ".join(
            f"{ra}/{cb}: {o} observed vs {e:.0f} expected ({z:+.1f})" for ra, cb, o, e, z in sorted(flagged, key=lambda x: -abs(x[4]))[:8]))


# ── survival helpers ──────────────────────────────────────────────────────────────────────────────────

def design(f, variables):
    """Numeric design matrix: age per decade, categorical dummies against the reference level, binary as is."""
    cols = {}
    for v in variables:
        if v == "age":
            cols["age/10y"] = f["age"] / 10
        elif v == "diagnosis_year":
            cols["year/10y"] = f["diagnosis_year"] / 10
        elif v in f.attrs["num"]:
            cols[v] = f[v].astype(float)
        else:
            s = f[v]; counts = s.value_counts(); ref = REFERENCE.get(v, counts.index[0])
            if ref not in counts.index:
                ref = counts.index[0]
            for lvl in counts.index:
                if lvl != ref and counts[lvl] >= MIN_LEVEL:
                    cols[f"{v}={lvl}"] = (s == lvl).astype(float).where(s.notna())
    return pd.DataFrame(cols, index=f.index)


def fit_cox(f, X):
    from lifelines import CoxPHFitter
    d = pd.concat([X, f[["os_years", "event"]]], axis=1).dropna()
    d = d[d["os_years"] > 0]
    d = d.loc[:, (d.nunique() > 1) | d.columns.isin(["os_years", "event"])]
    if len(d) < 50 or d["event"].sum() < 10:
        return None
    for pen in (0.0, 0.01, 0.1):
        try:
            return CoxPHFitter(penalizer=pen).fit(d, "os_years", "event"), len(d)
        except Exception:
            continue
    return None


def km_at(times, events, horizon):
    from lifelines import KaplanMeierFitter
    km = KaplanMeierFitter().fit(times, events)
    return float(km.predict(horizon)), int((times >= horizon).sum())


def interaction_lrt(f, a, b):
    Xa, Xb = design(f, [a]), design(f, [b])
    if Xa.shape[1] == 0 or Xb.shape[1] == 0:
        return None
    base = pd.concat([Xa, Xb], axis=1)
    prod = pd.DataFrame({f"{ca}*{cb}": Xa[ca] * Xb[cb] for ca in Xa for cb in Xb})
    m0, m1 = fit_cox(f, base), fit_cox(f, pd.concat([base, prod], axis=1))
    if not m0 or not m1 or m0[1] != m1[1]:
        return None
    k = len(m1[0].params_) - len(m0[0].params_)
    chi2 = 2 * (m1[0].log_likelihood_ - m0[0].log_likelihood_)
    return float(chi2), int(k), float(stats.chi2.sf(max(chi2, 0), max(k, 1))), m0[1]


# ── report ────────────────────────────────────────────────────────────────────────────────────────────

def overview(f, name, lines):
    n, ev = len(f), int(f["event"].sum())
    from lifelines import KaplanMeierFitter
    fu = float(KaplanMeierFitter().fit(f["os_years"], ~f["event"]).median_survival_time_)
    lines += [f"{n} patients, {ev} deaths ({pct(ev / n)}), median follow-up {fu:.1f} years (reverse Kaplan-Meier)"]
    if "diagnosis_year" in f and f["diagnosis_year"].notna().any():
        lines.append(f"diagnosed {int(f['diagnosis_year'].min())} - {int(f['diagnosis_year'].max())}")
    lines += ["", "variable            observed  missing  summary"]
    for c in f.attrs["num"]:
        x = f[c].dropna()
        lines.append(f"  {c:18s}{len(x):7d}  {pct(1 - len(x) / n):>7s}  mean {x.mean():.1f}, sd {x.std():.1f}, median {x.median():.1f}, IQR {x.quantile(.25):.1f}-{x.quantile(.75):.1f}")
    for c in f.attrs["cat"]:
        x = f[c].dropna()
        lines.append(f"  {c:18s}{len(x):7d}  {pct(1 - len(x) / n):>7s}  " + ", ".join(f"{k} {pct(v / len(x))}" for k, v in x.value_counts().head(7).items()))
    dead = [c for c in CATEGORICAL + NUMERIC if c not in f.attrs["cat"] + f.attrs["num"] and c in f]
    if dead:
        lines.append(f"  not usable (all missing or constant): {', '.join(dead)}")
    a = f["age"].dropna()
    kde = stats.gaussian_kde(a); xs = np.linspace(a.min(), a.max(), 400)
    from scipy.signal import find_peaks
    peaks = xs[find_peaks(kde(xs), prominence=0.05 * kde(xs).max())[0]]
    lines += ["", "AGE", f"  {hist(a)}", f"  skew {stats.skew(a):+.2f}, excess kurtosis {stats.kurtosis(a):+.2f}; Shapiro p={stats.shapiro(a.sample(min(len(a), 5000), random_state=0)).pvalue:.2g}; "
              f"density peaks at {', '.join(f'{p:.0f}' for p in peaks)}"]
    for c in ("sex", "era"):
        if c in f.attrs["cat"]:
            lines.append(f"  by {c}: " + "; ".join(f"{k}: n={len(g)}, mean {g.mean():.1f}, sd {g.std():.1f}" for k, g in f.groupby(c)["age"] if len(g) >= 10))


def pair_section(f, lines):
    pairs = all_pairs(f)
    lines += ["", "ALL PAIRS OF VARIABLES (effect size: Cramer's V / eta / |rho|, 0 = none, 1 = perfect; q = Benjamini-Hochberg over all pairs)",
              "  strongest pairs that are not true by definition:"]
    real = pairs[~pairs.structural]
    for _, r in real[(real.q_BH < 0.05)].head(25).iterrows():
        lines.append(f"    {r.a} ~ {r.b}: {r.measure} {r.effect:.2f} (n={r.n}, q={r.q_BH:.2g})")
    weak = real[(real.q_BH < 0.05) & (real.effect < 0.1)]
    lines.append(f"  ({len(real)} pairs tested, {int((real.q_BH < 0.05).sum())} with q<0.05, of which {len(weak)} detectable but weak, effect < 0.1)")
    big = real[real.effect >= 0.2]
    lines.append("  pairs with effect >= 0.2: " + (", ".join(f"{r.a}~{r.b} {r.effect:.2f}" for _, r in big.iterrows()) or "none"))
    return pairs


def combos_section(f, lines):
    lines += ["", "CHOSEN COMBINATIONS"]
    demo = [("age_band", "sex"), ("era", "sex"), ("era", "age_band")]
    clin = [("age_band", "stage_group"), ("sex", "stage_group"), ("sex", "pt_group"), ("sex", "pn_group"), ("sex", "grade_group"), ("age_band", "grade_group"),
            ("age_band", "pn_group"), ("era", "stage_group"), ("era", "grade_group"), ("era", "pn_group"), ("location", "sex"), ("location", "age_band"),
            ("mmr", "age_band"), ("mmr", "location"), ("mmr", "sex"), ("braf", "location"), ("braf", "mmr"), ("grade_group", "stage_group"), ("neoadjuvant", "age_band"),
            ("neoadjuvant", "location"), ("neoadjuvant", "stage_group"), ("era", "location"), ("era", "mmr")]
    for a, b in demo:
        if a in f.attrs["cat"] and b in f.attrs["cat"]:
            cross_table(f, a, b, lines, always=True)
    for a, b in clin:
        if a in f.attrs["cat"] and b in f.attrs["cat"]:
            cross_table(f, a, b, lines)
    if "age" in f.attrs["num"] and "sex" in f.attrs["cat"]:
        d = f[["age", "male"]].dropna()
        if len(d) > 50:
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(C=1e6).fit((d[["age"]] - 65) / 10, d["male"])
            lines.append(f"  male share per decade of age (logistic): odds ratio {np.exp(lr.coef_[0][0]):.2f}")


def logit_wald(X, y):
    """Unpenalised logistic regression with Wald standard errors: (coefficients, standard errors, p-values) incl. the intercept."""
    from sklearn.linear_model import LogisticRegression
    Z = np.column_stack([np.ones(len(X)), X.values.astype(float)])
    lr = LogisticRegression(C=1e8, max_iter=1000, fit_intercept=False).fit(Z, y)
    beta = lr.coef_[0]; p = 1 / (1 + np.exp(-Z @ beta))
    cov = np.linalg.pinv(Z.T @ (Z * (p * (1 - p))[:, None]))
    se = np.sqrt(np.diag(cov))
    return beta, se, 2 * stats.norm.sf(np.abs(beta / se))


def adjusted_section(f, lines):
    """Each binary property regressed on age, sex and era together: who is where, once the other demographics are accounted for."""
    lines += ["", "WHO IS WHERE, ADJUSTED (logistic regression on age band (reference 60-69), male, era together; odds ratios with 95% CI for terms with p<0.1)"]
    targets = [("stage_late", "stage III-IV"), ("grade_high", "high grade"), ("pt4", "pT4"), ("pn_positive", "node-positive"), ("right_sided", "right-sided"), ("dmmr", "dMMR")]
    X = design(f, ["age_band", "sex"] + (["era"] if "era" in f.attrs["cat"] else []))
    for col, label in targets:
        if f[col].notna().sum() < 100 or f[col].nunique() < 2:
            continue
        d = pd.concat([X, f[col].rename("y")], axis=1).dropna()
        d = d.loc[:, d.nunique() > 1]
        if d["y"].sum() < 20 or len(d) < 100:
            continue
        names = [c for c in d.columns if c != "y"]
        try:
            beta, se, p = logit_wald(d[names], d["y"].values)
        except Exception:
            continue
        hits = [f"{c} OR {np.exp(b):.2f} [{np.exp(b - 1.96 * s):.2f}, {np.exp(b + 1.96 * s):.2f}] p={pp:.2g}" for c, b, s, pp in zip(names, beta[1:], se[1:], p[1:]) if pp < 0.1]
        lines.append(f"  {label:14s} (n={len(d)}, {pct(d['y'].mean())}): " + ("; ".join(hits) or "no term at p<0.1"))


def survival_section(f, lines):
    lines += ["", "SURVIVAL IN COMBINATIONS"]
    if "age_band" in f.attrs["cat"] and "sex" in f.attrs["cat"]:
        lines.append("  5-year overall survival (Kaplan-Meier), n in the cell, cells with fewer than 20 patients left out:")
        d = f.dropna(subset=["age_band", "sex"]); bands = [b for b in ["<50", "50-59", "60-69", "70-79", "80+"] if b in d["age_band"].unique()]
        lines.append("    " + f"{'':8s}" + "".join(f"{s:>22s}" for s in sorted(d["sex"].unique())))
        for b in bands:
            row = []
            for s in sorted(d["sex"].unique()):
                g = d[(d["age_band"] == b) & (d["sex"] == s)]
                row.append(f"{km_at(g['os_years'], g['event'], 5)[0]:.0%} (n={len(g)}, d={int(g['event'].sum())})" if len(g) >= 20 else "-")
            lines.append("    " + f"{b:8s}" + "".join(f"{c:>22s}" for c in row))
    pairs = [("age", "sex"), ("age", "stage_group"), ("sex", "stage_group"), ("age", "grade_group"), ("sex", "grade_group"), ("grade_group", "stage_group"),
             ("era", "stage_group"), ("era", "sex"), ("age", "era"), ("age", "location"), ("sex", "location"), ("age", "mmr"), ("age", "pn_group"), ("sex", "pn_group")]
    res = []
    for a, b in pairs:
        if all(v in f.attrs["cat"] + f.attrs["num"] for v in (a, b)):
            r = interaction_lrt(f, a, b)
            if r:
                res.append((a, b, *r))
    if res:
        q = bh([r[4] for r in res])
        lines += ["  interaction tests (Cox likelihood-ratio: does the effect of one variable differ by the level of the other?):"]
        lines += [f"    {a} x {b}: chi2={c:.1f} on {k} df, p={p:.2g}, BH q={qq:.2g} (n={n})" + ("   <- interaction" if qq < 0.05 else "") for (a, b, c, k, p, n), qq in zip(res, q)]
    # confounding: univariable against adjusted
    cand = [v for v in ["age", "sex", "era", "stage_group", "grade_group", "location", "mmr"] if v in f.attrs["cat"] + f.attrs["num"]]
    full = fit_cox(f, design(f, cand))
    if full:
        multi, n_multi = full
        lines += ["", f"  UNIVARIABLE vs ADJUSTED hazard ratios (adjusted: all of {', '.join(cand)} together, n={n_multi}); a change of more than 25 % (and more than 0.15) in the log hazard ratio marks confounding"]
        X = design(f, cand)
        for col in X:
            uni = fit_cox(f, X[[c for c in X if c.split("=")[0] == col.split("=")[0]]])
            if not uni or col not in uni[0].params_ or col not in multi.params_:
                continue
            lu, la = uni[0].params_[col], multi.params_[col]
            shift = (la - lu) / abs(lu) if abs(lu) > 1e-9 else np.nan
            lines.append(f"    {col:22s} univariable HR {np.exp(lu):.2f}  adjusted HR {np.exp(la):.2f} [{np.exp(multi.confidence_intervals_.at[col, '95% lower-bound']):.2f}, "
                         f"{np.exp(multi.confidence_intervals_.at[col, '95% upper-bound']):.2f}]" + ("   <- changes with adjustment" if abs(shift) > 0.25 and abs(la - lu) > 0.15 else ""))


def missing_section(f, lines):
    lines += ["", "MISSINGNESS (is a value missing at random? p from Mann-Whitney / chi2 of missing vs observed; only variables with 2-98 % missing)"]
    n = len(f); any_found = False
    for c in f.attrs["cat"] + f.attrs["num"]:
        miss = f[c].isna()
        if not 0.02 < miss.mean() < 0.98:
            continue
        any_found = True; parts = []
        for o in [x for x in ("age", "male", "diagnosis_year", "event") if x != c and x in f]:
            d = pd.DataFrame({"m": miss, "o": f[o].astype(float)}).dropna()
            if d["m"].nunique() < 2:
                continue
            if o in ("male", "event"):
                t = pd.crosstab(d["m"], d["o"])
                if t.shape == (2, 2):
                    p = stats.chi2_contingency(t)[1]; parts.append((p, f"{o}: {pct(d.loc[d['m'], 'o'].mean())} among missing vs {pct(d.loc[~d['m'], 'o'].mean())} among observed (p={p:.2g})"))
            else:
                p = stats.mannwhitneyu(d.loc[d["m"], "o"], d.loc[~d["m"], "o"]).pvalue
                parts.append((p, f"{o}: median {d.loc[d['m'], 'o'].median():.0f} among missing vs {d.loc[~d['m'], 'o'].median():.0f} among observed (p={p:.2g})"))
        lines.append(f"  {c}: {pct(miss.mean())} missing; " + ("; ".join(t for p, t in parts if p < 0.01) or "no relation to age, sex, year or outcome at p<0.01"))
    if not any_found:
        lines.append("  no variable with partial missingness")


def young_section(f, lines):
    if "young" not in f or f["young"].sum() < 20:
        return
    y, o = f[f["young"] == 1], f[f["young"] == 0]
    lines += ["", f"THE YOUNG (age < 50: n={len(y)}) AGAINST THE REST (n={len(o)})"]
    for c in [c for c in ["sex", "grade_group", "stage_group", "pt_group", "pn_group", "location", "mmr", "braf", "morphology_group", "neoadjuvant", "era"] if c in f.attrs["cat"]]:
        d = f[[c, "young"]].dropna()
        v, p, n = cramers_v(d[c], d["young"])
        top = []
        for lvl in sorted(d[c].unique()):
            a, b = (d.loc[d["young"] == 1, c] == lvl).mean(), (d.loc[d["young"] == 0, c] == lvl).mean()
            top.append((abs(a - b), f"{lvl} {pct(a)} vs {pct(b)}"))
        lines.append(f"  {c:18s} V={v:.2f} p={p:.2g}: " + "; ".join(t for _, t in sorted(top, reverse=True)[:4]))
    lines.append(f"  deaths: {pct(y['event'].mean())} among the young vs {pct(o['event'].mean())} (follow-up differs, see the survival rows above)")


# ── between cohorts ───────────────────────────────────────────────────────────────────────────────────

def compare(frames, lines):
    (n1, f1), (n2, f2) = list(frames.items())[:2]
    shared = [c for c in ["age", "sex", "pt_group", "pn_group", "m_group", "grade_group", "stage_group"] if f1[c].notna().sum() > 50 and f2[c].notna().sum() > 50]
    lines += ["", f"{n1} AGAINST {n2}: shared variables {', '.join(shared)}", "  standardised mean difference (SMD; |SMD| > 0.1 = noticeable, > 0.3 = large):"]
    for c in shared:
        if c == "age":
            a, b = f1[c].dropna(), f2[c].dropna()
            lines.append(f"    age: {a.mean():.1f} vs {b.mean():.1f} years, SMD {(a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2):+.2f}")
        else:
            parts = []
            for lvl in sorted(set(f1[c].dropna()) | set(f2[c].dropna())):
                p1, p2 = (f1[c].dropna() == lvl).mean(), (f2[c].dropna() == lvl).mean()
                smd = (p1 - p2) / np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2 + 1e-12)
                parts.append(f"{lvl} {pct(p1)} vs {pct(p2)} ({smd:+.2f})")
            lines.append(f"    {c}: " + "; ".join(parts))
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    both = pd.concat([f1[shared].assign(y=0), f2[shared].assign(y=1)], ignore_index=True)
    X = both.drop(columns="y").copy(); X["age"] = X["age"].fillna(X["age"].median())
    X = pd.get_dummies(X, columns=[c for c in shared if c != "age"], dummy_na=True, dtype=float)
    X = (X - X.mean()) / (X.std() + 1e-9)
    p = cross_val_predict(LogisticRegression(max_iter=2000), X, both["y"], cv=5, method="predict_proba")[:, 1]
    lines.append(f"  can a model tell the cohorts apart from these variables alone? cross-validated AUC {roc_auc_score(both['y'], p):.2f} (0.5 = indistinguishable)")


# ── figures ───────────────────────────────────────────────────────────────────────────────────────────

def figures(frames, pair_tables, out):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6})
    for name, f in frames.items():
        cols = f.attrs["cat"] + f.attrs["num"]
        M = pd.DataFrame(np.nan, index=cols, columns=cols)
        for _, r in pair_tables[name].iterrows():
            if not r.structural:
                M.at[r.a, r.b] = M.at[r.b, r.a] = r.effect
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(M.values.astype(float), cmap="Blues", vmin=0, vmax=0.6); ax.grid(False)
        ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right"); ax.set_yticks(range(len(cols)), cols)
        for i, j in itertools.product(range(len(cols)), repeat=2):
            if i < j and not np.isnan(M.values[i, j]) and M.values[i, j] >= 0.1:
                ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if M.values[i, j] > 0.35 else "black")
        fig.colorbar(im, ax=ax, shrink=0.7, label="association (V / eta / |rho|)")
        ax.set_title(f"{name}: association between every pair of variables (blank = by definition or untestable)", fontsize=9)
        fig.tight_layout(); fig.savefig(f"{out}/{name}_associations.png", dpi=150); plt.close(fig)

        panels = [c for c in ("sex", "stage_group", "grade_group") if c in f.attrs["cat"]]
        if "age_band" in f.attrs["cat"] and panels:
            fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6), sharey=True)
            for ax, c in zip(np.atleast_1d(axes), panels):
                d = f[["age_band", c]].dropna(); t = pd.crosstab(d["age_band"], d[c], normalize="index") * 100
                t = t.reindex([b for b in ["<50", "50-59", "60-69", "70-79", "80+"] if b in t.index])
                bottom = np.zeros(len(t))
                for k, lvl in enumerate(t.columns):
                    ax.bar(t.index, t[lvl], bottom=bottom, color=OKABE[k % len(OKABE)], label=str(lvl), edgecolor="white", linewidth=1.5); bottom += t[lvl].values
                ax.set_title(f"{c} within age band", fontsize=9); ax.legend(fontsize=7, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.12))
            np.atleast_1d(axes)[0].set_ylabel("% of age band"); fig.suptitle(f"{name}: composition by age band", fontsize=10)
            fig.tight_layout(); fig.savefig(f"{out}/{name}_age_composition.png", dpi=150); plt.close(fig)
        if "era" in f.attrs["cat"] and f["era"].nunique() > 1:
            panels = [c for c in ("sex", "age_band", "stage_group") if c in f.attrs["cat"]]
            fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6), sharey=True)
            for ax, c in zip(np.atleast_1d(axes), panels):
                d = f[["era", c]].dropna(); t = pd.crosstab(d["era"], d[c], normalize="index") * 100
                t = t.reindex([e for e in ["<1990", "1990s", "2000s", "2010+"] if e in t.index]); bottom = np.zeros(len(t))
                for k, lvl in enumerate(t.columns):
                    ax.bar(t.index, t[lvl], bottom=bottom, color=OKABE[k % len(OKABE)], label=str(lvl), edgecolor="white", linewidth=1.5); bottom += t[lvl].values
                ax.set_title(f"{c} within era", fontsize=9); ax.legend(fontsize=7, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.12))
            np.atleast_1d(axes)[0].set_ylabel("% of era"); fig.suptitle(f"{name}: composition by diagnosis era", fontsize=10)
            fig.tight_layout(); fig.savefig(f"{out}/{name}_era_composition.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", action="append", required=True, help="NAME=patients.parquet path or mlflow-artifacts URI")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    frames = {}
    for spec in a.cohort:
        name, uri = spec.split("=", 1); frames[name] = prepare(load(uri))
    lines, pair_tables = [], {}
    for name, f in frames.items():
        lines += ["=" * 110, f"COHORT {name}", "=" * 110]
        overview(f, name, lines)
        pair_tables[name] = pair_section(f, lines)
        combos_section(f, lines)
        adjusted_section(f, lines)
        survival_section(f, lines)
        missing_section(f, lines)
        young_section(f, lines)
        lines.append("")
    if len(frames) >= 2:
        lines += ["=" * 110, "BETWEEN COHORTS", "=" * 110]
        compare(frames, lines)
    figures(frames, pair_tables, a.out)
    for name, t in pair_tables.items():
        t.to_csv(f"{a.out}/{name}_pairs.csv", index=False)
    open(f"{a.out}/cohort_analysis.txt", "w").write("\n".join(lines))
    print("\n".join(lines), flush=True)
    if os.environ.get("MLFLOW_TRACKING_URI"):
        import mlflow
        mlflow.set_experiment("RI Scale")
        with mlflow.start_run(run_name="Cohort analysis " + " + ".join(frames)):
            mlflow.log_artifacts(a.out)


if __name__ == "__main__":
    main()
