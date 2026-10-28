"""Apply the tissue-classification linear probe (MLflow experiment 104, run 0e2230c7...) to pre-extracted Virchow2 tile features.

The model is ``Linear(2560 -> 7)`` + softmax on ``concat([feats, average_patch_feats])`` of the raw Virchow2 embeddings (no
normalisation), exported by the run as ``onnx/linear_head.onnx``; its weights are read from ``tissue_head.npz`` (W, b, class names).
The ``thresholds`` of the run are annotation-coverage rules used to LABEL the training tiles, not inference settings: here every tile
gets the full probability vector and its most likely class, as in the repository's own prediction config.

Per slide it writes ``<out>/<slide>.npz`` with ``coords`` (N, 2) float32, ``probs`` (N, 7) float16 and ``pred`` (N,) uint8, and one line
into ``<out>/summary-<shard>.jsonl`` (tiles, share of every predicted class, mean probabilities, mean confidence). Resumable; slides are
only read.

usage: apply_tissue_model.py --features <dir> --head tissue_head.npz --out <dir> [--slide-list names.txt] [--shard i/n] [--limit N]
"""
import argparse
import json
import os
import time

import h5py
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--features", required=True); ap.add_argument("--head", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--slide-list", default=""); ap.add_argument("--shard", default="0/1"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--chunk", type=int, default=100_000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

head = np.load(a.head)
W, b, classes = torch.as_tensor(head["W"], device=dev), torch.as_tensor(head["b"], device=dev), [str(c) for c in head["classes"]]
assert W.shape == (7, 2560) and len(classes) == 7, W.shape

names = sorted(f[:-3] for f in os.listdir(a.features) if f.endswith(".h5"))
if a.slide_list:
    wanted = {l.strip().rsplit(".", 1)[0] for l in open(a.slide_list) if l.strip()}
    names = [n for n in names if n in wanted]
si, sn = map(int, a.shard.split("/")); names = names[si::sn]
names = [n for n in names if not os.path.exists(f"{a.out}/{n}.npz")]
names = names[: a.limit] if a.limit else names
print(f"{len(names)} slides to do on {dev}", flush=True)


@torch.no_grad()
def predict(feats, avg):
    out = np.empty((len(feats), 7), np.float32)
    for i in range(0, len(feats), a.chunk):
        x = torch.cat([torch.as_tensor(feats[i:i + a.chunk], device=dev), torch.as_tensor(avg[i:i + a.chunk], device=dev)], 1)
        out[i:i + a.chunk] = torch.softmax(x @ W.T + b, 1).cpu().numpy()
    return out


t0 = time.time(); n_tiles = 0
with open(f"{a.out}/summary-{si}of{sn}.jsonl", "a") as summary:
    for k, name in enumerate(names):
        try:
            with h5py.File(f"{a.features}/{name}.h5", "r") as h:
                if str(h.attrs.get("extraction_complete", "")) != "True":
                    summary.write(json.dumps(dict(slide=name, ok=False, error="extraction incomplete")) + "\n"); continue
                feats, avg, coords = h["data/feats"][:], h["data/average_patch_feats"][:], h["patch_grid/patch_coords"][:]
            probs = predict(feats, avg); pred = probs.argmax(1).astype(np.uint8)
            np.savez(f"{a.out}/{name}.npz", coords=coords.astype(np.float32), probs=probs.astype(np.float16), pred=pred)
            share = np.bincount(pred, minlength=7) / len(pred)
            summary.write(json.dumps(dict(slide=name, ok=True, tiles=int(len(pred)), share=dict(zip(classes, share.round(5).tolist())),
                                          mean_prob=dict(zip(classes, probs.mean(0).round(5).tolist())),
                                          mean_confidence=float(probs.max(1).mean()))) + "\n"); summary.flush()
            n_tiles += len(pred)
        except Exception as ex:
            summary.write(json.dumps(dict(slide=name, ok=False, error=repr(ex)[:200])) + "\n")
        if (k + 1) % 100 == 0 or k + 1 == len(names):
            print(f"{k + 1}/{len(names)} slides, {n_tiles:,} tiles, {n_tiles / (time.time() - t0):,.0f} tiles/s", flush=True)
print("done", flush=True)
