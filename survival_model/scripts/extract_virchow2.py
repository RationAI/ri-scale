"""Virchow2 features of the segmented lymph nodes (+ a margin) of the MMCI survival cohort.

Per slide: the 256-px tiles at 0.5 um/px that overlap the predicted node polygons dilated by --margin-um are read
in place (OpenSlide, read-only), embedded with Virchow2 and written to <out>/<slide>.h5:
    feats   (N, 2560) float16   [class token (1280) | mean of the patch tokens (1280)]
    coords  (N, 2)    int32     level-0 x, y of the tile's top-left corner
    inside  (N,)      float16   fraction of the tile covered by the (undilated) node polygons
    node    (N,)      int16     index of the node (feature order in the GeoJSON) the tile belongs to
Resumable: slides with an existing .h5 are skipped. Nothing but the .h5 files is written.

usage: extract_virchow2.py --cohort cohort_slides.json --out <dir> [--count-only] [--limit N] [--shard i/n]
"""
import argparse, json, math, os, time
from concurrent.futures import ThreadPoolExecutor

import cv2, h5py, numpy as np, openslide

SLIDES = "/data/fs201316/rh02261/data/mmci_lymph_nodes"  # read-only
GEO = os.path.expanduser("~/sam-ft/predictions/mmci_sam3-sem-v1/geojson")
MS = 16            # the masks are built at 1/16 of level 0
TILE_UM = 128.0    # 256 px at 0.5 um/px
INPUT_PX = 224     # Virchow2 input size (a 256-px tile is resized to it, as in its own transform)

ap = argparse.ArgumentParser()
ap.add_argument("--cohort", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--margin-um", type=float, default=50.0); ap.add_argument("--min-overlap", type=float, default=0.25)
ap.add_argument("--count-only", action="store_true"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--shard", default="0/1"); ap.add_argument("--readers", type=int, default=16); ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--slides", default="", help="comma-separated slide ids, overrides the cohort list")
a = ap.parse_args()


def rings(geometry):
    polys = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
    return polys


def draw(geometry, mask):
    for poly in rings(geometry):
        for k, ring in enumerate(poly):
            pts = (np.asarray(ring, dtype=np.float64) / MS).round().astype(np.int32)
            cv2.fillPoly(mask, [pts], 1 if k == 0 else 0)


def plan(sid):
    """Tiles of one slide: (coords, inside, node, tile_l0, mpp)."""
    sl = openslide.OpenSlide(f"{SLIDES}/{sid}.mrxs")
    W, H = sl.dimensions; mpp = float(sl.properties["openslide.mpp-x"]); sl.close()
    tile_l0 = TILE_UM / mpp
    nodes = json.load(open(f"{GEO}/{sid}.geojson"))["features"]
    hm, wm = math.ceil(H / MS), math.ceil(W / MS)
    r = int(round(a.margin_um / mpp / MS)); kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    inside = np.zeros((hm, wm), np.uint8); lab_in = np.zeros((hm, wm), np.int16); lab_d = np.zeros((hm, wm), np.int16)
    for i, f in enumerate(nodes):
        m = np.zeros((hm, wm), np.uint8); draw(f["geometry"], m)
        inside |= m; lab_in[(m > 0) & (lab_in == 0)] = i + 1
        ys, xs = np.nonzero(m)
        if len(ys) == 0: continue
        y0, y1, x0, x1 = max(0, ys.min() - r), min(hm, ys.max() + r + 1), max(0, xs.min() - r), min(wm, xs.max() + r + 1)
        d = cv2.dilate(m[y0:y1, x0:x1], kern); sub = lab_d[y0:y1, x0:x1]; sub[(d > 0) & (sub == 0)] = i + 1
    lab_d[lab_in > 0] = lab_in[lab_in > 0]  # a node's own pixels win over a neighbour's margin
    dil = lab_d > 0

    nx, ny = math.ceil(W / tile_l0), math.ceil(H / tile_l0)
    gx = np.clip(np.round(np.arange(nx + 1) * tile_l0 / MS).astype(int), 0, wm)
    gy = np.clip(np.round(np.arange(ny + 1) * tile_l0 / MS).astype(int), 0, hm)
    def frac(mask):
        I = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        s = I[np.ix_(gy[1:], gx[1:])] - I[np.ix_(gy[:-1], gx[1:])] - I[np.ix_(gy[1:], gx[:-1])] + I[np.ix_(gy[:-1], gx[:-1])]
        area = np.outer(np.diff(gy), np.diff(gx)).clip(min=1)
        return s / area
    f_dil, f_in = frac(dil), frac(inside > 0)
    jj, ii = np.nonzero(f_dil >= a.min_overlap)
    coords = np.stack([np.round(ii * tile_l0), np.round(jj * tile_l0)], 1).astype(np.int32)
    node = np.empty(len(ii), np.int16)
    for t, (j, i) in enumerate(zip(jj, ii)):
        v = lab_d[gy[j]:gy[j + 1], gx[i]:gx[i + 1]]; v = v[v > 0]
        node[t] = np.bincount(v).argmax() - 1 if len(v) else -1
    return coords, f_in[jj, ii].astype(np.float16), node, tile_l0, mpp


cohort = json.load(open(a.cohort))
slides = a.slides.split(",") if a.slides else sorted(s for v in cohort.values() for s in v)
si, sn = map(int, a.shard.split("/")); slides = slides[si::sn]
if a.limit: slides = slides[: a.limit]
os.makedirs(a.out, exist_ok=True)
todo = [s for s in slides if a.count_only or not os.path.exists(f"{a.out}/{s}.h5")]
print(f"{len(slides)} slides, {len(todo)} to do", flush=True)

if a.count_only:
    total = 0; t0 = time.time()
    for k, s in enumerate(todo):
        try: n = len(plan(s)[0])
        except Exception as e: print("FAIL", s, repr(e)[:150], flush=True); continue
        total += n
        if k % 200 == 0: print(k, total, f"{time.time() - t0:.0f}s", flush=True)
    print("tiles", total, "-> fp16 2560-d GB:", round(total * 2560 * 2 / 1e9, 1)); raise SystemExit

import timm, torch
from PIL import Image
from timm.layers import SwiGLUPacked

model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU).cuda().eval()
cfg = timm.data.resolve_data_config(model.pretrained_cfg, model=model)
mean = torch.tensor(cfg["mean"], device="cuda").view(1, 3, 1, 1); std = torch.tensor(cfg["std"], device="cuda").view(1, 3, 1, 1)
assert cfg["input_size"][1:] == (INPUT_PX, INPUT_PX), cfg


def read_tile(args):
    sl, (x, y), lv, size = args
    rg = sl.read_region((int(x), int(y)), lv, (size, size))
    bg = Image.new("RGB", rg.size, (255, 255, 255)); bg.paste(rg, mask=rg.getchannel("A"))
    return cv2.resize(np.asarray(bg), (INPUT_PX, INPUT_PX), interpolation=cv2.INTER_AREA)


@torch.no_grad()
def embed(batch):
    x = (torch.from_numpy(np.stack(batch)).cuda().permute(0, 3, 1, 2).float() / 255 - mean) / std
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(x)
    return torch.cat([out[:, 0], out[:, 5:].mean(1)], 1).half().cpu().numpy()  # class token | mean of patch tokens (4 registers skipped)


pool = ThreadPoolExecutor(a.readers)
t_start = time.time(); n_done = n_tiles = 0
for s in todo:
    t0 = time.time()
    try:
        coords, inside, node, tile_l0, mpp = plan(s)
        sl = openslide.OpenSlide(f"{SLIDES}/{s}.mrxs")
        lv = sl.get_best_level_for_downsample(tile_l0 / 256); size = int(round(tile_l0 / sl.level_downsamples[lv]))
        feats = np.empty((len(coords), 2560), np.float16)
        chunk = a.batch * 2
        for i in range(0, len(coords), chunk):
            tiles = list(pool.map(read_tile, [(sl, c, lv, size) for c in coords[i:i + chunk]]))
            for j in range(0, len(tiles), a.batch):
                feats[i + j:i + j + len(tiles[j:j + a.batch])] = embed(tiles[j:j + a.batch])
        sl.close()
        tmp = f"{a.out}/{s}.h5.tmp"
        with h5py.File(tmp, "w") as h:
            h.create_dataset("feats", data=feats, compression="lzf"); h.create_dataset("coords", data=coords)
            h.create_dataset("inside", data=inside); h.create_dataset("node", data=node)
            h.attrs.update(slide=s, mpp=mpp, tile_l0=tile_l0, margin_um=a.margin_um, tile_px=256, input_px=INPUT_PX,
                           layout="class_token(1280)|patch_mean(1280)", model="paige-ai/Virchow2")
        os.replace(tmp, f"{a.out}/{s}.h5")
        n_done += 1; n_tiles += len(coords)
        print(f"{s} {len(coords)} tiles {time.time() - t0:.1f}s | total {n_done} slides {n_tiles} tiles {n_tiles / (time.time() - t_start):.0f} tiles/s", flush=True)
    except Exception as e:
        print("FAIL", s, repr(e)[:200], flush=True)
