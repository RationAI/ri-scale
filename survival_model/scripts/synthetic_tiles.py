"""Synthetic tile store + node table for testing the tile-level models without patient data.

Each patient gets twice as many nodes as ``nodes_examined`` (serial sections) and twice ``nodes_positive`` of them are tumour:
a quarter of the tiles of a tumour node is shifted along one hidden direction, so the true lymph-node ratio is known.

usage: python scripts/synthetic_tiles.py <patients.parquet> <out_dir> [--dim 128]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("patients", type=Path); ap.add_argument("out", type=Path)
ap.add_argument("--dim", type=int, default=128); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--shift", type=float, default=2.0, help="shift of a tumour tile in units of the noise s.d.")
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(a.seed)
p = pd.read_parquet(a.patients).reset_index(drop=True).sort_values("patient_id").reset_index(drop=True)
w = rng.normal(size=a.dim); w /= np.linalg.norm(w)
X, node, offsets, nodes_rows, core = [], [], [0], [], []
for _, q in p.iterrows():
    n_nodes = max(3, 2 * (int(q.nodes_examined) if q.nodes_examined > 0 else 15))
    n_tum = min(n_nodes, int(round(2 * q.nodes_positive))) if q.nodes_positive == q.nodes_positive else 0
    is_tum = np.zeros(n_nodes, bool); is_tum[rng.permutation(n_nodes)[:n_tum]] = True
    for k in range(n_nodes):
        n_tiles = int(rng.integers(40, 250)); t = rng.normal(size=(n_tiles, a.dim))
        if is_tum[k]:
            t[rng.permutation(n_tiles)[: n_tiles // 4]] += a.shift * w
        X.append(t.astype(np.float16)); node.append(np.full(n_tiles, (k % 5) * 10000 + k, np.int32))
        nodes_rows.append((q.patient_id, k % 5, k, n_tiles, 0, n_tiles * 0.0164)); core.append(rng.normal(size=8))
    offsets.append(offsets[-1] + sum(len(x) for x in X[-n_nodes:]))
np.savez(a.out / "tiles.npz", X=np.concatenate(X), node=np.concatenate(node), offsets=np.array(offsets, np.int64), patients=p["patient_id"].to_numpy().astype(str))
c = list(zip(*nodes_rows))
np.savez(a.out / "nodes.npz", patient=np.array(c[0]), slide_idx=np.array(c[1]), node=np.array(c[2]), n_core=np.array(c[3]), n_margin=np.array(c[4]),
         area_mm2=np.array(c[5]), core=np.array(core, np.float16), margin=np.array(core, np.float16))
print(f"{len(p)} patients, {offsets[-1]} tiles -> {a.out}")
