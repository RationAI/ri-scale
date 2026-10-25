"""Synthetic node embeddings + patient table for testing ``image_survival`` without patient data.

Known ground truth: a patient's risk grows with the share of "tumour-like" nodes (embeddings shifted along one hidden
direction) and mildly with age; sex has no effect. The signal is spread over many noisy dimensions, as in real embeddings.

usage: python scripts/synthetic_nodes.py <out_dir> [--patients 214] [--dim 2560] [--seed 0]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path); ap.add_argument("--patients", type=int, default=214)
ap.add_argument("--dim", type=int, default=2560); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--signal", type=float, default=1.0, help="strength of the image effect on the log hazard")
a = ap.parse_args()
rng = np.random.default_rng(a.seed); a.out.mkdir(parents=True, exist_ok=True)

direction = rng.normal(size=a.dim); direction /= np.linalg.norm(direction)
rows, core, margin = [], [], []
patients = []
for p in range(a.patients):
    pid = f"p{p:04d}"; n = int(np.clip(rng.lognormal(3.3, 0.6), 3, 150))
    share = rng.beta(0.6, 2.0)                                   # fraction of tumour-like nodes
    tumour = rng.random(n) < share
    emb = rng.normal(size=(n, a.dim)) + 0.12 * np.outer(tumour, direction * np.sqrt(a.dim))
    age = float(np.clip(rng.normal(68, 11), 30, 95)); male = rng.random() < 0.55
    log_h = a.signal * 2.5 * share + 0.03 * (age - 68)
    t = rng.exponential(1 / (0.06 * np.exp(log_h))); c = rng.uniform(2, 11)
    patients.append({"patient_id": pid, "case_ids": f"2015_{p:05d}", "age": age, "sex": "M" if male else "F",
                     "time_days": min(t, c) * 365.25, "os_event": bool(t <= c), "css_event": bool(t <= c)})
    for k in range(n):
        area = float(rng.lognormal(1.2, 0.8)); n_core = max(1, int(area / 0.0164))
        rows.append((pid, k % 6, k, n_core, int(n_core * 0.2), area)); core.append(emb[k]); margin.append(emb[k] + rng.normal(size=a.dim) * 0.5)
cols = list(zip(*rows))
np.savez(a.out / "nodes.npz", patient=np.array(cols[0]), slide_idx=np.array(cols[1]), node=np.array(cols[2]), n_core=np.array(cols[3]),
         n_margin=np.array(cols[4]), area_mm2=np.array(cols[5]), core=np.array(core, np.float16), margin=np.array(margin, np.float16))
pd.DataFrame(patients).to_csv(a.out / "patients_min.csv", index=False)
print(f"{len(patients)} patients, {len(core)} nodes, {sum(p['os_event'] for p in patients)} events -> {a.out}")
