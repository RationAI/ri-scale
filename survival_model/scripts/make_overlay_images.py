"""Image overlays (RGBA PNG) for the xOpat viewer: one picture per slide that covers the slide's extent exactly.

xOpat stacks a mask image on a slide by normalised position, so the picture only has to have the slide's aspect ratio:
pixel (X, Y) of a picture with ``W * k / tile`` columns stands for the level-0 region [X, X + 1) * tile / k.
Pixels without data are transparent; colours are baked in (shown with the ``identity`` shader).

  tissue  <pred dir>/<slide>.npz (apply_tissue_model.py)   -> <out>/<slide>.tissue.png   one colour per class, k = 4 pixels per tile
  nodes   <geojson dir>/<slide>.geojson (SAM3 predictions) -> <out>/<slide>.nodes.png    teal fill, 1 pixel = ``--node-um`` micrometres

usage: make_overlay_images.py tissue --src <dir> --slides-dir <dir with the slides> --out <dir> [--slides list.txt] [--ext .mrxs]
       make_overlay_images.py nodes  --src <dir> --slides-dir <dir with the slides> --out <dir> [--slides list.txt]
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["Nerve", "Blood", "Connective-Tissue", "Fat", "Epithelium", "Muscle", "Other"]
COLOURS = {"Nerve": (42, 120, 214), "Blood": (235, 104, 52), "Connective-Tissue": (27, 175, 122), "Fat": (237, 161, 0),
           "Epithelium": (232, 123, 164), "Muscle": (0, 131, 0), "Other": (98, 80, 214)}
TISSUE_ALPHA, NODE_ALPHA, NODE_COLOUR = 170, 150, (31, 181, 143)


def slide_geometry(path):
    """(width, height, um per pixel) of level 0, read from the slide header only."""
    import openslide
    s = openslide.OpenSlide(path)
    w, h = s.dimensions
    mpp = float(s.properties.get("openslide.mpp-x") or 0.2428)
    s.close()
    return w, h, mpp


def tissue_image(npz_path, width, height, per_tile=4):
    z = np.load(npz_path)
    coords, pred = z["coords"].astype(np.float64), z["pred"]
    dx = np.diff(np.sort(coords[:, 0])); dx = dx[(dx > 400) & (dx < 700)]
    step = float(np.median(dx)) if len(dx) else 527.2
    p = step / per_tile                                         # level-0 pixels per picture pixel
    wi, hi = int(round(width / p)), int(round(height / p))
    img = np.zeros((hi, wi, 4), np.uint8)
    x0, y0 = np.rint(coords[:, 0] / p).astype(np.int64), np.rint(coords[:, 1] / p).astype(np.int64)
    rgba = np.array([COLOURS[c] + (TISSUE_ALPHA,) for c in CLASSES], np.uint8)[pred]
    for j in range(per_tile):
        for i in range(per_tile):
            yy, xx = np.clip(y0 + j, 0, hi - 1), np.clip(x0 + i, 0, wi - 1)
            img[yy, xx] = rgba
    return Image.fromarray(img, "RGBA")


def nodes_image(geojson_path, width, height, mpp, node_um=16.0):
    p = node_um / mpp                                           # level-0 pixels per picture pixel
    wi, hi = max(1, int(round(width / p))), max(1, int(round(height / p)))
    im = Image.new("RGBA", (wi, hi), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    for f in json.load(open(geojson_path))["features"]:
        g = f["geometry"]; polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for poly in polys:
            for k, ring in enumerate(poly):
                pts = [(x / p, y / p) for x, y in ring]
                if len(pts) >= 3:
                    draw.polygon(pts, fill=NODE_COLOUR + (NODE_ALPHA,) if k == 0 else (0, 0, 0, 0))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tissue", "nodes"]); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--slides-dir", default=""); ap.add_argument("--ext", default=".mrxs"); ap.add_argument("--slides", default="")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--node-um", type=float, default=16.0)
    ap.add_argument("--fake-dims", default="", help="WxH, for tests without slides")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    src_ext = ".npz" if a.mode == "tissue" else ".geojson"
    names = sorted(f[:-len(src_ext)] for f in os.listdir(a.src) if f.endswith(src_ext))
    if a.slides:
        wanted = {l.strip().rsplit(".", 1)[0] for l in open(a.slides) if l.strip()}; names = [n for n in names if n in wanted]
    names = names[: a.limit] if a.limit else names
    done = 0
    for k, name in enumerate(names):
        try:
            if a.fake_dims:
                w, h = map(int, a.fake_dims.split("x")); mpp = 0.2428
            else:
                w, h, mpp = slide_geometry(f"{a.slides_dir}/{name}{a.ext}")
            im = tissue_image(f"{a.src}/{name}.npz", w, h) if a.mode == "tissue" else nodes_image(f"{a.src}/{name}.geojson", w, h, mpp, a.node_um)
            im.save(f"{a.out}/{name}.{a.mode}.png", optimize=True); done += 1
        except Exception as e:
            print("FAIL", name, repr(e)[:150], flush=True)
        if (k + 1) % 50 == 0 or k + 1 == len(names):
            print(f"{k + 1}/{len(names)} slides, {done} written", flush=True)


if __name__ == "__main__":
    main()
