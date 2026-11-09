"""Which node profiles are repeated cuts of the same lymph node?

A node is cut several times (serial sections on one slide, or the same block cut again), so one node can appear as several profiles and the
counts, sizes and shapes of node_morphology.py are then pseudo-replicated. Two profiles of one node have nearly the same outline, so every pair
of profiles of a patient is compared by outline overlap (IoU after centring, 36 rotations; and without rotation). A pair counts as a repeat only
if the overlap beats what UNRELATED nodes of the same size reach: the null is pairs from different patients, matched on the area ratio.

  1. calibration: IoU of pairs from different patients -> 99 % and 99.9 % thresholds per area-ratio bin
  2. how many same-slide and same-patient/other-slide pairs exceed them, against the 1 % expected by chance
  3. groups: linked profiles (union-find) = one node cut several times, at a strict (p99.9) and a liberal (p99) level; node_groups_*.csv
  4. features of node_morphology.py again with ONE profile per group (the largest) -> <out>/dedup_p999/ and <out>/dedup_p99/

usage: python -m image_survival.node_duplicates --geojson <dir> --cohort cohort_slides.json --patients-min patients_min.csv --out <dir>
       [--link-across-slides]   then:  python -m image_survival.node_morphology --patients ... --out <out>/dedup_p999 --steps analyse
"""
import argparse
import csv
import json
import os
import warnings
from collections import Counter

import cv2
import numpy as np
import pandas as pd

from image_survival.node_morphology import patient_features, profile_rows

warnings.filterwarnings("ignore")
UM_PX, SIZE = 50.0, 128                      # raster: 50 um per pixel, 6.4 mm square, centred on the profile
ANGLES = np.arange(0, 360, 10)
RATIO_EDGES = [1.0, 1.3, 1.7, 2.5, 4.0]      # larger / smaller area; pairs beyond 4 are never called repeats
N_NULL = 10


def raster(outline_um):
    pts = np.concatenate(outline_um)
    c = pts.mean(0)
    img = np.zeros((SIZE, SIZE), np.uint8)
    for p in outline_um:
        cv2.fillPoly(img, [np.round((p - c) / UM_PX + SIZE / 2).astype(np.int32)], 1)
    return img.astype(np.float32)


def rotations(img):
    """(len(ANGLES), SIZE*SIZE) rotated copies of one mask."""
    out = []
    for ang in ANGLES:
        m = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), float(ang), 1.0)
        out.append(cv2.warpAffine(img, m, (SIZE, SIZE), flags=cv2.INTER_LINEAR).ravel())
    return np.stack(out)


def iou_to(rot, area, others, other_area):
    """IoU of one profile (its rotations) against many others: returns (best over rotations, without rotation)."""
    inter = rot @ others.T
    iou = inter / (rot.sum(1, keepdims=True) + other_area[None] - inter + 1e-9)
    return iou.max(0), iou[0]


def load(a):
    with open(a.patients_min) as f:
        patient_of = {r["case_ids"]: r["patient_id"] for r in csv.DictReader(f)}
    with open(a.cohort) as f:
        cohort = json.load(f)
    rows = []
    for case, slides in sorted(cohort.items()):
        for slide in sorted(slides):
            path = f"{a.geojson}/{slide}.geojson"
            if not os.path.exists(path):
                continue
            for k, r in enumerate(profile_rows(path, with_outline=True)):
                rows.append({"patient_id": patient_of[case], "slide": slide, "slide_idx": sorted(slides).index(slide), "idx": k, **r})
    return rows


def features_of_kept(rows, pid, slide, rep):
    """node_morphology features per patient from the kept profiles (indices ``rep``) only."""
    feats = []
    for p in sorted(set(pid)):
        sl = sorted(set(slide[pid == p])); mine = [i for i in rep if pid[i] == p]
        per = [(sum(slide[i] == s for i in mine), sum(rows[i]["area"] for i in mine if slide[i] == s)) for s in sl]
        feats.append(patient_features(p, per, [rows[i] for i in mine]))
    return pd.DataFrame(feats)


def bin_of(ratio):
    return np.digitize(ratio, RATIO_EDGES[1:-1]) if np.all(np.isfinite(ratio)) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geojson", required=True); ap.add_argument("--cohort", required=True); ap.add_argument("--patients-min", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--link-across-slides", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rows = load(a)
    n = len(rows); pid = np.array([r["patient_id"] for r in rows]); slide = np.array([r["slide"] for r in rows])
    masks = np.stack([raster(r["outline_um"]) .ravel() for r in rows]); area = masks.sum(1)
    rng = np.random.default_rng(0)
    print(f"{n} profiles, {len(set(pid))} patients, {len(set(slide))} slides", flush=True)

    # 1. calibration on pairs from different patients
    null_best, null_flat, null_ratio = [], [], []
    for i in range(n):
        ratio = np.maximum(area, area[i]) / np.maximum(np.minimum(area, area[i]), 1)
        cand = np.flatnonzero((pid != pid[i]) & (ratio <= RATIO_EDGES[-1]))
        if len(cand) == 0:
            continue
        j = rng.choice(cand, min(N_NULL, len(cand)), replace=False)
        b, f = iou_to(rotations(masks[i].reshape(SIZE, SIZE)), area[i], masks[j], area[j])
        null_best += list(b); null_flat += list(f); null_ratio += list(ratio[j])
    null_best, null_flat, null_ratio = map(np.asarray, (null_best, null_flat, null_ratio))
    nb = np.digitize(null_ratio, RATIO_EDGES[1:-1])
    thr = {q: {"best": [np.quantile(null_best[nb == k], q) for k in range(len(RATIO_EDGES) - 1)],
               "flat": [np.quantile(null_flat[nb == k], q) for k in range(len(RATIO_EDGES) - 1)]} for q in (0.99, 0.999)}
    lines = [f"{n} profiles of {len(set(pid))} patients in {len(set(slide))} slides; null = {len(null_best)} pairs from different patients",
             "", "NULL (unrelated nodes of similar size): IoU thresholds by area ratio larger/smaller",
             "  ratio        " + "  ".join(f"{RATIO_EDGES[k]:.1f}-{RATIO_EDGES[k + 1]:.1f}" for k in range(len(RATIO_EDGES) - 1))]
    for q in (0.99, 0.999):
        for m in ("best", "flat"):
            lines.append(f"  {q:.1%} {m:5s}  " + "  ".join(f"{v:7.2f}" for v in thr[q][m]))

    # 2. all pairs within a patient
    pairs = []
    for p in sorted(set(pid)):
        ids = np.flatnonzero(pid == p)
        for pos, i in enumerate(ids[:-1]):
            j = ids[pos + 1:]
            b, f = iou_to(rotations(masks[i].reshape(SIZE, SIZE)), area[i], masks[j], area[j])
            ratio = np.maximum(area[j], area[i]) / np.maximum(np.minimum(area[j], area[i]), 1)
            for jj, bb, ff, rr in zip(j, b, f, ratio):
                pairs.append((i, jj, bb, ff, rr, slide[i] == slide[jj]))
    P = pd.DataFrame(pairs, columns=["i", "j", "iou_best", "iou_flat", "ratio", "same_slide"])
    P = P[P.ratio <= RATIO_EDGES[-1]].copy(); P["bin"] = np.digitize(P.ratio, RATIO_EDGES[1:-1])
    for q, tag in ((0.99, "p99"), (0.999, "p999")):
        P[tag] = (P.iou_best >= np.array(thr[q]["best"])[P.bin]) | (P.iou_flat >= np.array(thr[q]["flat"])[P.bin])
    lines += ["", "PAIRS OF THE SAME PATIENT (area ratio <= 4): share above the null thresholds; chance alone gives 2 % (p99, either test) and 0.2 % (p99.9)"]
    for name, sub in (("same slide", P[P.same_slide]), ("other slide of the patient", P[~P.same_slide])):
        r99, r999 = sub.p99.mean(), sub.p999.mean()
        lines.append(f"  {name:28s} {len(sub):6d} pairs | above p99: {sub.p99.sum():5d} ({r99:.1%}) | above p99.9: {sub.p999.sum():4d} ({r999:.2%})"
                     f" | repeats beyond chance about {max(0, sub.p99.sum() - 0.02 * len(sub)):.0f} pairs")

    # 3. groups and 4. one profile per node, at a strict (p99.9) and a liberal (p99) linking level
    old = pd.read_csv(f"{a.out}/features.csv").set_index("patient_id")
    for level, tag in (("p999", "strict, p99.9"), ("p99", "liberal, p99")):
        link = P[P[level] & (P.same_slide | a.link_across_slides)]
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for i, j in zip(link.i, link.j):
            parent[find(i)] = find(j)
        group = np.array([find(i) for i in range(n)])
        gsize = Counter(group)
        chance = (0.001 if level == "p999" else 0.01) * 2 * len(P[P.same_slide | a.link_across_slides])
        lines += ["", f"GROUPS, {tag} ({'incl. other slides' if a.link_across_slides else 'same slide only'}): {n} profiles -> {len(gsize)} nodes "
                  f"({1 - len(gsize) / n:.1%} fewer); links {len(link)}, of which about {chance:.0f} expected by chance alone",
                  "  profiles per node: " + "  ".join(f"{k}:{v}" for k, v in sorted(Counter(gsize.values()).items()))]
        per_slide = pd.DataFrame({"slide": slide, "group": group}).groupby("slide").agg(profiles=("group", "size"), nodes=("group", "nunique"))
        lines.append("  slides by profiles on the slide -> median distinct nodes (n slides): " + "; ".join(
            f"{k}: {per_slide[per_slide.profiles == k].nodes.median():.0f} ({(per_slide.profiles == k).sum()})" for k in range(2, 9) if (per_slide.profiles == k).any()))
        keep = {}
        for i in range(n):
            g = group[i]
            if g not in keep or area[i] > area[keep[g]]:
                keep[g] = i
        rep = sorted(keep.values()); repset = set(rep)
        os.makedirs(f"{a.out}/dedup_{level}", exist_ok=True)
        pd.DataFrame({"patient_id": pid, "slide": slide, "profile": [r["idx"] for r in rows], "group": group, "group_size": [gsize[g] for g in group],
                      "kept": [i in repset for i in range(n)]}).to_csv(f"{a.out}/node_groups_{level}.csv", index=False)
        feats = features_of_kept(rows, pid, slide, rep)
        F = feats; F.to_csv(f"{a.out}/dedup_{level}/features.csv", index=False)
        o = old.reindex(F.patient_id)
        lines += [f"  features with one profile per node -> {a.out}/dedup_{level}/features.csv",
                  f"  per patient (median): nodes {o.nodes_total.median():.0f} -> {F.nodes_total.median():.0f}; mean profile area {o.area_mean_mm2.median():.2f} -> {F.area_mean_mm2.median():.2f} mm2"]

    # a few pairs to look at: overlay of the outlines after the best rotation (red/blue); stays on the cluster, outlines only
    try:
        top = P[P.p999 & P.same_slide].sort_values("iou_best", ascending=False)
        mid = P[P.p999 & P.same_slide].sort_values("iou_best").head(8)
        tiles = []
        for _, r in pd.concat([top.head(8), mid]).iterrows():
            A, B = masks[int(r.i)].reshape(SIZE, SIZE), masks[int(r.j)].reshape(SIZE, SIZE)
            ious = [(np.minimum(rot.reshape(SIZE, SIZE), B).sum() / np.maximum(rot.reshape(SIZE, SIZE), B).sum(), k) for k, rot in enumerate(rotations(A))]
            Ar = rotations(A)[max(ious)[1]].reshape(SIZE, SIZE)
            tile = np.stack([B, np.zeros_like(B), Ar], -1); tiles.append(np.clip(tile * 255, 0, 255).astype(np.uint8))
        if tiles:
            sheet = np.vstack([np.hstack(tiles[k:k + 8]) for k in range(0, len(tiles), 8)])
            cv2.imwrite(f"{a.out}/duplicate_pairs.png", sheet); lines.append(f"overlay of pairs (top row: highest overlap, second row: just above the threshold) -> {a.out}/duplicate_pairs.png")
    except Exception as e:                                 # the figure is a convenience
        lines.append(f"overlay figure skipped: {e}")
    P.drop(columns=["bin"]).to_csv(f"{a.out}/pairs.csv", index=False)
    open(f"{a.out}/duplicates.txt", "w").write("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
