"""Are two node profiles cuts of the same lymph node? Second, independent signal: the tissue itself (Virchow2 node embeddings).

node_duplicates.py compares OUTLINES. Here every pair of profiles of a patient is also compared by TISSUE: cosine similarity of the node
embeddings (mean Virchow2 embedding of the core tiles, every dimension standardised over the cohort first). Two cuts of one node should look
alike inside (same cell composition, same follicles), unrelated nodes should not.

  1. does the tissue signal agree with the outline signal?  Among pairs on the same slide, tissue similarity of the pairs the outline calls
     repeats against the others (AUC; 0.5 = no relation). Same slide on both sides, so the staining of the slide cancels out.
  2. a calibrated tissue call: similarity above the 95th / 99th percentile of pairs on the same slide that the outline says are different
  3. confident repeats = outline AND tissue agree; grouped (union-find), one profile per node (the largest); features and survival analysis
     as in node_duplicates.py -> <out>/dedup_combined/features.csv
  4. the same comparison for profiles on different slides of one patient

usage: python -m image_survival.node_tissue_match --geojson <dir> --cohort cohort_slides.json --patients-min patients_min.csv \
           --nodes nodes.npz --out <dir of node_duplicates.py, with pairs.csv and features.csv>
"""
import argparse
import os
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from image_survival.node_duplicates import features_of_kept, load

warnings.filterwarnings("ignore")


def auc(pos, neg):
    """P(a random positive scores above a random negative), ties half."""
    from scipy.stats import rankdata
    x = np.concatenate([pos, neg]); r = rankdata(x)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geojson", required=True); ap.add_argument("--cohort", required=True); ap.add_argument("--patients-min", required=True)
    ap.add_argument("--nodes", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = load(a); n = len(rows)
    P = pd.read_csv(f"{a.out}/pairs.csv")
    pid = np.array([r["patient_id"] for r in rows]); slide = np.array([r["slide"] for r in rows]); area = np.array([r["area"] for r in rows])

    z = np.load(a.nodes); key = {(p, int(s), int(k)): m for m, (p, s, k) in enumerate(zip(z["patient"], z["slide_idx"], z["node"]))}
    emb_row = np.array([key.get((r["patient_id"], r["slide_idx"], r["feature_idx"]), -1) for r in rows])
    have = emb_row >= 0
    E = np.full((n, z["core"].shape[1]), np.nan, np.float32); E[have] = z["core"][emb_row[have]].astype(np.float32)
    mu, sd = np.nanmean(E, 0), np.nanstd(E, 0) + 1e-6
    E = (E - mu) / sd; E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    lines = [f"{n} profiles, {int(have.sum())} with a tissue embedding"]

    ok = have[P.i.values] & have[P.j.values]; P = P[ok].copy()
    P["tissue"] = np.concatenate([(E[P.i.values[k:k + 50000]] * E[P.j.values[k:k + 50000]]).sum(1) for k in range(0, len(P), 50000)])
    same, other = P[P.same_slide], P[~P.same_slide]
    lines += ["", f"tissue similarity (cosine, standardised embeddings): median {same.tissue.median():.2f} for pairs on the same slide, "
                  f"{other.tissue.median():.2f} for pairs on different slides of a patient (the gap is the slide / staining effect, so only same-slide pairs are compared)"]

    # 1. agreement of the two signals on the same slide
    strict, band, rest = same[same.p999], same[same.p99 & ~same.p999], same[~same.p99]
    lines += ["", "1. DO TISSUE AND OUTLINE AGREE? (pairs on the same slide)",
              f"   tissue similarity, median: outline repeat strict {strict.tissue.median():.2f} (n={len(strict)}) | borderline {band.tissue.median():.2f} (n={len(band)}) | outline says different {rest.tissue.median():.2f} (n={len(rest)})",
              f"   AUC of tissue similarity: strict repeats vs different {auc(strict.tissue.values, rest.tissue.values):.3f} | borderline vs different {auc(band.tissue.values, rest.tissue.values):.3f}  (0.5 = unrelated, 1 = perfect)"]
    # the outline signal depends on the area ratio, tissue similarity might too: repeat the AUC within area-ratio classes
    cls = [(1.0, 1.3), (1.3, 1.7), (1.7, 4.01)]
    lines.append("   AUC strict vs different, by area ratio: " + "; ".join(
        f"{lo:.1f}-{min(hi, 4.0):.1f}: {auc(strict[(strict.ratio >= lo) & (strict.ratio < hi)].tissue.values, rest[(rest.ratio >= lo) & (rest.ratio < hi)].tissue.values):.3f}"
        for lo, hi in cls if ((strict.ratio >= lo) & (strict.ratio < hi)).sum() > 10))

    # 2. calibrated tissue call
    thr95, thr99 = np.quantile(rest.tissue, 0.95), np.quantile(rest.tissue, 0.99)
    P["t95"], P["t99"] = P.tissue >= thr95, P.tissue >= thr99
    same = P[P.same_slide]
    lines += ["", f"2. TISSUE CALL: similarity >= {thr95:.2f} (95th percentile of same-slide pairs the outline calls different) or >= {thr99:.2f} (99th)",
              f"   same-slide pairs with tissue >= p95: outline repeat strict {strict.tissue.ge(thr95).mean():.0%}, borderline {band.tissue.ge(thr95).mean():.0%}, different {rest.tissue.ge(thr95).mean():.0%} (5 % by construction)",
              f"   same-slide pairs with tissue >= p99: strict {strict.tissue.ge(thr99).mean():.0%}, borderline {band.tissue.ge(thr99).mean():.0%}, different {rest.tissue.ge(thr99).mean():.0%} (1 % by construction)"]

    # 3. confident repeats and groups
    P["confident"] = P.p99 & P.t95 & P.same_slide
    P["confident_strict"] = P.p999 & P.t95 & P.same_slide
    both = int(((same.p99) & (same.t95)).sum()); exp = same.p99.mean() * same.t95.mean() * len(same)
    lines += ["", f"3. CONFIDENT REPEATS (outline >= p99 AND tissue >= p95): {int(P.confident.sum())} pairs (strict outline p99.9 AND tissue p95: {int(P.confident_strict.sum())}); "
                  f"if the two signals were unrelated only {exp:.0f} would be expected"]
    for level, tag in (("confident", "outline p99 + tissue p95"), ("confident_strict", "outline p99.9 + tissue p95")):
        link = P[P[level]]; parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for i, j in zip(link.i, link.j):
            parent[find(i)] = find(j)
        group = np.array([find(i) for i in range(n)]); gsize = Counter(group)
        keep = {}
        for i in range(n):
            if group[i] not in keep or area[i] > area[keep[group[i]]]:
                keep[group[i]] = i
        rep = sorted(keep.values()); repset = set(rep)
        os.makedirs(f"{a.out}/dedup_{level}", exist_ok=True)
        pd.DataFrame({"patient_id": pid, "slide": slide, "profile": [r["idx"] for r in rows], "group": group, "group_size": [gsize[g] for g in group],
                      "kept": [i in repset for i in range(n)]}).to_csv(f"{a.out}/node_groups_{level}.csv", index=False)
        F = features_of_kept(rows, pid, slide, rep); F.to_csv(f"{a.out}/dedup_{level}/features.csv", index=False)
        lines += [f"   {tag}: {n} profiles -> {len(gsize)} nodes ({1 - len(gsize) / n:.1%} fewer); profiles per node: " + "  ".join(f"{k}:{v}" for k, v in sorted(Counter(gsize.values()).items())),
                  f"   slides by profiles -> median distinct nodes: " + "; ".join(
                      f"{k}: {pd.DataFrame({'s': slide, 'g': group}).groupby('s').g.nunique()[pd.Series(slide).value_counts()[lambda v: v == k].index].median():.0f}" for k in (2, 3, 4, 5, 6)
                      if (pd.Series(slide).value_counts() == k).any()),
                  f"   features -> {a.out}/dedup_{level}/features.csv"]

    # 4. different slides of one patient
    oth = P[~P.same_slide]
    flagged, plain = oth[oth.p99], oth[~oth.p99]
    thr95_o = np.quantile(plain.tissue, 0.95)
    lines += ["", "4. PROFILES ON DIFFERENT SLIDES OF ONE PATIENT",
              f"   tissue similarity, median: outline p99 {flagged.tissue.median():.2f} (n={len(flagged)}) vs not flagged {plain.tissue.median():.2f} (n={len(plain)}); AUC {auc(flagged.tissue.values, plain.tissue.values):.3f}",
              f"   share of pairs with tissue above the 95th percentile of the unflagged: outline-flagged {flagged.tissue.ge(thr95_o).mean():.1%} (5 % if unrelated)",
              f"   very high tissue similarity (>= {np.quantile(oth.tissue, 0.999):.2f}, top 0.1 % of all different-slide pairs): {int((oth.tissue >= np.quantile(oth.tissue, 0.999)).sum())} pairs, of which {int(oth[oth.tissue >= np.quantile(oth.tissue, 0.999)].p99.sum())} also pass the outline p99"]
    P.to_csv(f"{a.out}/pairs_tissue.csv", index=False)
    open(f"{a.out}/tissue_match.txt", "w").write("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
