"""Node-level embeddings from the per-slide Virchow2 files written by ``scripts/extract_virchow2.py``.

One row per segmented lymph node: the mean embedding of its core tiles (``inside >= --core``) and of its margin
tiles, plus its size. Slides are identified only by a running index per patient, so nothing downstream can see
the slide-name suffix (LNN / LNP / M, a pathologist's staging result).

usage: python -m image_survival.pool_nodes --features <dir> --cohort cohort_slides.json --patients patients_min.csv --out nodes.npz
"""
import argparse, csv, json, os

import h5py
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True); ap.add_argument("--cohort", required=True)
    ap.add_argument("--patients", required=True, help="csv with patient_id and case_ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--core", type=float, default=0.5, help="a tile is a core tile when this fraction of it is node")
    ap.add_argument("--margin", type=float, default=0.05, help="a tile is a margin tile below this node fraction")
    a = ap.parse_args()

    patient_of = {r["case_ids"]: r["patient_id"] for r in csv.DictReader(open(a.patients))}
    cohort = json.load(open(a.cohort))
    rows: dict[str, list] = {k: [] for k in ("patient", "slide_idx", "node", "n_core", "n_margin", "area_mm2")}
    core, margin = [], []
    skipped = {"no_file": 0, "no_core_tile": 0}
    for case, slides in sorted(cohort.items()):
        for k, slide in enumerate(sorted(slides)):
            path = f"{a.features}/{slide}.h5"
            if not os.path.exists(path):
                skipped["no_file"] += 1; continue
            with h5py.File(path) as h:
                feats, inside, node = h["feats"][:], h["inside"][:].astype(np.float32), h["node"][:]
                tile_mm2 = float(h.attrs["tile_l0"] * h.attrs["mpp"] / 1000) ** 2
            for n in np.unique(node[node >= 0]):
                mine = node == n
                is_core, is_margin = mine & (inside >= a.core), mine & (inside < a.margin)
                if not is_core.any():
                    skipped["no_core_tile"] += 1; continue
                rows["patient"].append(patient_of[case]); rows["slide_idx"].append(k); rows["node"].append(int(n))
                rows["n_core"].append(int(is_core.sum())); rows["n_margin"].append(int(is_margin.sum()))
                rows["area_mm2"].append(float(inside[mine].sum() * tile_mm2))
                core.append(feats[is_core].astype(np.float32).mean(0).astype(np.float16))
                margin.append(feats[is_margin].astype(np.float32).mean(0).astype(np.float16) if is_margin.any()
                              else np.full(feats.shape[1], np.nan, np.float16))
    np.savez(a.out, **{k: np.asarray(v) for k, v in rows.items()}, core=np.stack(core), margin=np.stack(margin))
    print(f"{len(core)} nodes of {len(set(rows['patient']))} patients; skipped {skipped}")


if __name__ == "__main__":
    main()
