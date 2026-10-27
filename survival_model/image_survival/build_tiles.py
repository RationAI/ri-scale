"""Tile store for the tile-level detector: every core tile of every node of the cohort, reduced to ``--dim`` dimensions.

The 2560-d Virchow2 embeddings of ``scripts/extract_virchow2.py`` (about 22 GB) are projected with a PCA fitted on a random
sample of tiles (unsupervised: no labels or outcomes are used) and whitened. Output ``tiles.npz``:
    X       (N, dim) float16   projected tiles, grouped by patient
    node    (N,)     int32     slide_index * 10000 + node index (nodes are unique within a patient)
    offsets (P + 1,) int64     tiles of patient i are X[offsets[i]:offsets[i + 1]]
    patients (P,)              patient ids

usage: python -m image_survival.build_tiles --features <dir> --cohort cohort_slides.json --patients patients_min.csv --out tiles.npz
"""
import argparse
import csv
import json
import os

import h5py
import numpy as np
from sklearn.decomposition import PCA


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True); ap.add_argument("--cohort", required=True)
    ap.add_argument("--patients", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--dim", type=int, default=128); ap.add_argument("--core", type=float, default=0.5)
    ap.add_argument("--sample", type=float, default=0.02, help="fraction of tiles used to fit the PCA")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    with open(a.patients) as f:
        patient_of = {r["case_ids"]: r["patient_id"] for r in csv.DictReader(f)}
    with open(a.cohort) as f:
        cohort = json.load(f)
    plan = []  # (patient, slide index, path)
    for case, slides in sorted(cohort.items(), key=lambda kv: patient_of[kv[0]]):
        for k, slide in enumerate(sorted(slides)):
            path = f"{a.features}/{slide}.h5"
            if os.path.exists(path):
                plan.append((patient_of[case], k, path))

    rng = np.random.default_rng(a.seed); sample = []
    for _, _, path in plan:   # pass 1: sample tiles for the PCA
        with h5py.File(path) as h:
            core = np.flatnonzero((h["inside"][:].astype(np.float32) >= a.core) & (h["node"][:] >= 0))
            pick = np.sort(rng.choice(core, max(1, int(len(core) * a.sample)), replace=False)) if len(core) else core
            if len(pick):
                sample.append(h["feats"][pick].astype(np.float32))
    sample = np.concatenate(sample)
    pca = PCA(a.dim, random_state=a.seed, whiten=True).fit(sample)
    print(f"PCA on {len(sample)} tiles, explained variance {pca.explained_variance_ratio_.sum():.3f}", flush=True)
    mean, comp, scale = pca.mean_.astype(np.float32), pca.components_.astype(np.float32), np.sqrt(pca.explained_variance_).astype(np.float32)

    X, node, owners = [], [], []
    for n, (patient, k, path) in enumerate(plan):   # pass 2: project every core tile
        with h5py.File(path) as h:
            keep = np.flatnonzero((h["inside"][:].astype(np.float32) >= a.core) & (h["node"][:] >= 0))
            if not len(keep):
                continue
            feats = h["feats"][keep].astype(np.float32)
            X.append((((feats - mean) @ comp.T) / scale).astype(np.float16))
            node.append(k * 10000 + h["node"][keep].astype(np.int32)); owners.append(np.full(len(keep), patient))
        if n % 200 == 0:
            print(f"{n}/{len(plan)} slides", flush=True)
    X, node, owners = np.concatenate(X), np.concatenate(node), np.concatenate(owners)
    patients, first = np.unique(owners, return_index=True)
    assert (np.sort(first) == first).all(), "tiles must be grouped by patient"
    offsets = np.append(first, len(owners)).astype(np.int64)
    np.savez(a.out, X=X, node=node, offsets=offsets, patients=patients)
    print(f"{len(X)} tiles of {len(patients)} patients -> {a.out}")


if __name__ == "__main__":
    main()
