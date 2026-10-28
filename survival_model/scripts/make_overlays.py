"""GeoJSON overlays (level-0 pixel coordinates, one file per slide) for the xOpat viewer: Annotations -> import.

Stock xOpat only draws single-part polygons reliably, so every multi-part object is split into separate Polygon features.

  sam     SAM3 node predictions (predictions/<run>/geojson/<slide>.geojson) -> <out>/<slide>.json
  tissue  tile classes of the tissue classifier (apply_tissue_model.py output, <slide>.npz) -> <out>/<slide>.tissue.json
          one feature per connected region of a class (merged 128 um tiles; holes kept as interior rings), with
          properties {"class", "color", "tiles", "mean_prob"}

usage: make_overlays.py sam --src <geojson dir> --out <overlay dir> [--slides list.txt]
       make_overlays.py tissue --src <dir with npz> --out <overlay dir> [--slides list.txt]
"""
import argparse
import json
import os

import numpy as np

CLASSES = ["Nerve", "Blood", "Connective-Tissue", "Fat", "Epithelium", "Muscle", "Other"]
COLOURS = {"Nerve": "#2a78d6", "Blood": "#eb6834", "Connective-Tissue": "#1baf7a", "Fat": "#eda100",
           "Epithelium": "#e87ba4", "Muscle": "#008300", "Other": "#6250d6"}


def split_parts(geometry):
    """Polygon / MultiPolygon -> list of Polygon coordinate lists."""
    return [geometry["coordinates"]] if geometry["type"] == "Polygon" else list(geometry["coordinates"])


def sam_overlay(src):
    fc = json.load(open(src)); feats = []
    for f in fc["features"]:
        for part in split_parts(f["geometry"]):
            feats.append({"type": "Feature", "properties": dict(f["properties"], part_of=f["properties"].get("instance_id")),
                          "geometry": {"type": "Polygon", "coordinates": part}})
    return {"type": "FeatureCollection", "features": feats}


def tissue_overlay(npz_path):
    from shapely.geometry import box
    from shapely.ops import unary_union
    z = np.load(npz_path); coords, pred, probs = z["coords"].astype(np.float64), z["pred"], z["probs"].astype(np.float32)
    dx = np.diff(np.sort(coords[:, 0])); dx = dx[(dx > 400) & (dx < 700)]
    step = float(np.median(dx)) if len(dx) else 527.2
    feats = []
    for k, name in enumerate(CLASSES):
        sel = np.flatnonzero(pred == k)
        if not len(sel):
            continue
        # tiles of this class merged into regions; coordinates are the tile top-left corners (level 0)
        merged = unary_union([box(x, y, x + step, y + step) for x, y in coords[sel]])
        for poly in (merged.geoms if merged.geom_type == "MultiPolygon" else [merged]):
            rings = [list(map(list, poly.exterior.coords))] + [list(map(list, r.coords)) for r in poly.interiors]
            inside = np.flatnonzero((coords[sel][:, 0] >= poly.bounds[0] - 1) & (coords[sel][:, 0] <= poly.bounds[2]) &
                                    (coords[sel][:, 1] >= poly.bounds[1] - 1) & (coords[sel][:, 1] <= poly.bounds[3]))
            feats.append({"type": "Feature",
                          "properties": {"class": name, "color": COLOURS[name], "tiles": int(round(poly.area / step ** 2)),
                                         "mean_prob": round(float(probs[sel[inside], k].mean()), 3) if len(inside) else None},
                          "geometry": {"type": "Polygon", "coordinates": [[[round(x, 1), round(y, 1)] for x, y in r] for r in rings]}})
    return {"type": "FeatureCollection", "features": feats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sam", "tissue"]); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--slides", default=""); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    ext = ".geojson" if a.mode == "sam" else ".npz"
    names = sorted(f[:-len(ext)] for f in os.listdir(a.src) if f.endswith(ext))
    if a.slides:
        wanted = {l.strip().rsplit(".", 1)[0] for l in open(a.slides) if l.strip()}; names = [n for n in names if n in wanted]
    names = names[: a.limit] if a.limit else names
    n_feat = 0
    for k, name in enumerate(names):
        oc = sam_overlay(f"{a.src}/{name}{ext}") if a.mode == "sam" else tissue_overlay(f"{a.src}/{name}{ext}")
        out = f"{a.out}/{name}{'' if a.mode == 'sam' else '.tissue'}.json"
        json.dump(oc, open(out, "w"), separators=(",", ":")); n_feat += len(oc["features"])
        if (k + 1) % 50 == 0 or k + 1 == len(names):
            print(f"{k + 1}/{len(names)} slides, {n_feat} features", flush=True)


if __name__ == "__main__":
    main()
