"""Are the sizes and shapes of lymph-node profiles normally distributed, or do they have peaks (separate groups)?

Node level (one value per predicted node profile of the survival cohort) and patient level (the per-patient summaries of node_morphology.py).
For every measure:
  * moments (skewness, excess kurtosis) and the largest gap to a fitted normal curve (Kolmogorov D, an effect size: with thousands of values a
    p-value rejects normality for any tiny deviation)
  * Gaussian mixtures with 1-4 components: BIC, and for the best one weights, means, s.d. and Ashman's D (D > 2 = clean separation of two groups)
  * kernel density: number of peaks (prominence >= 5 % of the highest), and Silverman's bootstrap test of "one peak" (small p = more than one peak)
  * a text histogram

usage: python -m image_survival.node_distributions --geojson <dir> --cohort cohort_slides.json --patients-min patients_min.csv \
           --features <features.csv of node_morphology> --out <dir>
"""
import argparse
import csv
import json
import os
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from scipy import ndimage, signal, stats
from sklearn.mixture import GaussianMixture

from image_survival.node_morphology import FEATURES, profile_rows

warnings.filterwarnings("ignore")
BLOCKS = " ▁▂▃▄▅▆▇█"


def text_histogram(x, bins=44):
    cnt, edges = np.histogram(x, bins=bins)
    top = cnt.max() or 1
    bar = "".join(BLOCKS[int(round(c / top * (len(BLOCKS) - 1)))] for c in cnt)
    return f"{edges[0]:.2f} {bar} {edges[-1]:.2f}"


def binned_kde(x, bw, lo, hi, bins=2048):
    hist, edges = np.histogram(x, bins=bins, range=(lo, hi))
    return ndimage.gaussian_filter1d(hist.astype(float), bw / (edges[1] - edges[0]), mode="constant")


def count_modes(dens, rel_prominence=0.0):
    if rel_prominence <= 0:
        d = dens[1:-1]
        return int(np.sum((d > dens[:-2]) & (d >= dens[2:]) & (d > 1e-9 * dens.max())))
    return len(signal.find_peaks(dens, prominence=rel_prominence * dens.max())[0])


def silverman_p(x, n_boot=200, seed=0):
    """Silverman (1981): critical bandwidth for one mode, p = share of smoothed bootstrap samples that need more than one mode at it."""
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    lo, hi = x.min() - 4 * x.std(), x.max() + 4 * x.std()
    a, b = 1e-3 * x.std(), 2 * x.std()
    for _ in range(25):                                   # smallest bandwidth with a single mode
        h = (a + b) / 2
        a, b = (a, h) if count_modes(binned_kde(x, h, lo, hi)) <= 1 else (h, b)
    hc = b
    var = x.var()
    above = 0
    for _ in range(n_boot):
        y = rng.choice(x, len(x)) + hc * rng.standard_normal(len(x))
        y = x.mean() + (y - x.mean()) / np.sqrt(1 + hc ** 2 / var)
        above += count_modes(binned_kde(y, hc, lo, hi)) > 1
    return float((above + 1) / (n_boot + 1)), float(hc / x.std())


def analyse_measure(x, name, level, back=None):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    z = (x - x.mean()) / x.std()
    row = {"level": level, "measure": name, "n": len(x), "skew": float(stats.skew(x)), "excess_kurtosis": float(stats.kurtosis(x)),
           "ks_D_vs_normal": float(stats.kstest(z, "norm").statistic)}
    Z = z.reshape(-1, 1); fits = [GaussianMixture(k, n_init=3, random_state=0).fit(Z) for k in (1, 2, 3, 4)]
    bic = [g.bic(Z) for g in fits]; best = int(np.argmin(bic)) + 1
    g = fits[best - 1]; order = np.argsort(g.means_.ravel())
    comps = [(float(g.weights_[i]), float(g.means_[i, 0] * x.std() + x.mean()), float(np.sqrt(g.covariances_[i, 0, 0]) * x.std())) for i in order]
    row.update(best_k=best, dBIC_best_vs_1=float(bic[0] - bic[best - 1]))
    if best >= 2:
        (w1, m1, s1), (w2, m2, s2) = comps[0], comps[-1]
        row["ashman_D_extreme_pair"] = float(np.sqrt(2) * abs(m2 - m1) / np.sqrt(s1 ** 2 + s2 ** 2))
    p, hc = silverman_p(x)
    dens = binned_kde(x, 0.9 * min(x.std(), stats.iqr(x) / 1.34) * len(x) ** -0.2, x.min() - 3 * x.std(), x.max() + 3 * x.std())
    row.update(kde_peaks=count_modes(dens, 0.05), silverman_p_unimodal=p)
    row["components"] = "; ".join(f"{w:.0%} at {m:.3g} (sd {s:.2g})" for w, m, s in comps)
    if back:
        row["components_back_transformed"] = "; ".join(f"{w:.0%} at {back(m):.3g}" for w, m, _ in comps)
    return row, text_histogram(x)


def node_table(a):
    with open(a.patients_min) as f:
        patient_of = {r["case_ids"]: r["patient_id"] for r in csv.DictReader(f)}
    with open(a.cohort) as f:
        cohort = json.load(f)
    rows, per_slide = [], []
    for case, slides in sorted(cohort.items()):
        for slide in sorted(slides):
            path = f"{a.geojson}/{slide}.geojson"
            if not os.path.exists(path):
                continue
            profiles = profile_rows(path); per_slide.append(len(profiles))
            rows += [{"patient_id": patient_of[case], **r} for r in profiles]
    return pd.DataFrame(rows), per_slide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geojson", required=True); ap.add_argument("--cohort", required=True); ap.add_argument("--patients-min", required=True)
    ap.add_argument("--features", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    nodes, per_slide = node_table(a)
    nodes.to_csv(f"{a.out}/node_profiles.csv", index=False)
    lines = [f"node profiles: {len(nodes)} in {len(per_slide)} slides of {nodes.patient_id.nunique()} patients",
             "smallest profile kept by the segmentation: about 0.18 mm2 (the area filter), so the left end of the size curve is cut off", ""]
    results = []
    node_measures = [("log area (ln mm2)", np.log(nodes["area"]), lambda m: float(np.exp(m))), ("solidity", nodes["solidity"], None),
                     ("elongation (short/long axis)", nodes["elong"], None), ("segmentation confidence", nodes["prob"], None)]
    for name, v, back in node_measures:
        row, hist = analyse_measure(v, name, "node", back); results.append(row)
        lines += [f"NODE LEVEL  {name}", f"  {hist}", f"  skew {row['skew']:+.2f}, excess kurtosis {row['excess_kurtosis']:+.2f}, largest gap to a normal curve D = {row['ks_D_vs_normal']:.3f}",
                  f"  best mixture: {row['best_k']} component(s) (BIC better than 1 by {row['dBIC_best_vs_1']:.0f}): {row['components']}"
                  + (f"  -> in mm2: {row['components_back_transformed']}" if "components_back_transformed" in row else ""),
                  f"  peaks in the density: {row['kde_peaks']} | Silverman test of one peak: p = {row['silverman_p_unimodal']:.3f}"
                  + (f" | Ashman D of the outer pair: {row['ashman_D_extreme_pair']:.2f}" if "ashman_D_extreme_pair" in row else ""), ""]
    counts = Counter(per_slide)
    lines += ["NODE PROFILES PER SLIDE (number of slides with k profiles):",
              "  " + "  ".join(f"{k}:{counts[k]}" for k in sorted(counts) if k <= 14) + f"  (more: {sum(v for k, v in counts.items() if k > 14)})", ""]
    feats = pd.read_csv(a.features)
    lines.append("PATIENT LEVEL (the summaries of node_morphology.py; log scale for count and area measures)")
    for name, (desc, log) in FEATURES.items():
        if name not in feats:
            continue
        v = feats[name].astype(float).dropna(); v = np.log1p(v) if log else v
        if v.std() > 0:
            row, hist = analyse_measure(v, name, "patient"); results.append(row)
            lines.append(f"  {name:24s} skew {row['skew']:+.2f} | D {row['ks_D_vs_normal']:.3f} | best k {row['best_k']} (dBIC {row['dBIC_best_vs_1']:.0f}) | peaks {row['kde_peaks']} | Silverman p {row['silverman_p_unimodal']:.3f}")
    pd.DataFrame(results).to_csv(f"{a.out}/distributions.csv", index=False)
    open(f"{a.out}/distributions.txt", "w").write("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
