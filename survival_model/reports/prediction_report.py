"""HTML report of the tissue-classification probe applied to a subset of slides (output of scripts/apply_tissue_model.py).

Shows the class composition per slide type, the confidence of the predictions and, for a gallery of slides, a map of the predicted
class and of the confidence over the tissue (one pixel per 128 um tile). Embedded PNG / SVG only, no slide pixels.

usage: prediction_report.py --pred <dir with <slide>.npz and summary-*.jsonl> --out report.html
"""
import argparse
import base64
import glob
import html
import io
import json
import os
import re
import time

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--per-type", type=int, default=4); ap.add_argument("--map-size", type=int, default=300)
ap.add_argument("--features-dir", default="", help="shown in the text only")
ap.add_argument("--viewer-url", default="http://127.0.0.1:9000/?slides=MMCI/{slide}.mrxs", help="link of a slide in xOpat ({slide} = slide id); empty = no links")
a = ap.parse_args()
e = html.escape
t0 = time.time()
CLASSES = ["Nerve", "Blood", "Connective-Tissue", "Fat", "Epithelium", "Muscle", "Other"]
COLOURS = {"Nerve": (42, 120, 214), "Blood": (235, 104, 52), "Connective-Tissue": (27, 175, 122), "Fat": (237, 161, 0),
           "Epithelium": (232, 123, 164), "Muscle": (0, 131, 0), "Other": (98, 80, 214)}  # fixed categorical order


def slide_type(name):
    m = re.search(r"-([A-Za-z]+)_?$", name)
    return m.group(1) if m else "(none)"


rows = []
for f in sorted(glob.glob(f"{a.pred}/summary-*.jsonl")):
    rows += [json.loads(l) for l in open(f) if l.strip()]
ok = {r["slide"]: r for r in rows if r.get("ok")}
failed = [r for r in rows if not r.get("ok")]
for r in ok.values():
    r["type"] = slide_type(r["slide"])
types = sorted({r["type"] for r in ok.values()}, key=lambda t: -sum(r["type"] == t for r in ok.values()))
total_tiles = sum(r["tiles"] for r in ok.values())
share_all = np.array([sum(r["share"][c] * r["tiles"] for r in ok.values()) / total_tiles for c in CLASSES])


def png_b64(arr):
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


RAMP = np.array([[205, 226, 251], [134, 182, 239], [57, 135, 229], [28, 92, 171], [13, 54, 107]], dtype=np.float32)


def ramp(v):
    p = np.clip(v, 0, 1) * (len(RAMP) - 1); i = np.minimum(p.astype(int), len(RAMP) - 2); w = (p - i)[:, None]
    return (RAMP[i] * (1 - w) + RAMP[i + 1] * w).astype(np.uint8)


def grid_map(x, y, colours, bg=(236, 235, 230)):
    dx = np.diff(np.sort(x)); dx = dx[(dx > 400) & (dx < 700)]
    step = float(np.median(dx)) if len(dx) else 527.2
    ix, iy = np.rint((x - x.min()) / step).astype(int), np.rint((y - y.min()) / step).astype(int)
    img = np.zeros((iy.max() + 1, ix.max() + 1, 3), np.uint8); img[:] = bg; img[iy, ix] = colours
    side = max(img.shape[:2]); im = Image.fromarray(img)
    if side <= a.map_size:
        s = max(1, a.map_size // side); im = im.resize((img.shape[1] * s, img.shape[0] * s), Image.NEAREST)
    else:
        s = a.map_size / side; im = im.resize((max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))), Image.BOX)
    return np.asarray(im), (img.shape[1], img.shape[0])


gallery = []
for t in types:
    pool = sorted([r for r in ok.values() if r["type"] == t and r["tiles"] >= 150], key=lambda r: r["tiles"])
    if not pool:
        continue
    for r in [pool[int(i)] for i in np.unique(np.linspace(0, len(pool) - 1, min(a.per_type, len(pool))).round())]:
        z = np.load(f"{a.pred}/{r['slide']}.npz"); pred, probs, c = z["pred"], z["probs"].astype(np.float32), z["coords"]
        m1, size = grid_map(c[:, 0], c[:, 1], np.array([COLOURS[CLASSES[k]] for k in range(7)], np.uint8)[pred])
        conf = probs.max(1); m2, _ = grid_map(c[:, 0], c[:, 1], ramp((conf - 0.4) / 0.6))
        gallery.append(dict(r=r, cls=png_b64(m1), conf=png_b64(m2), grid=size))

CSS = """
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--b:#2a78d6}
@media (prefers-color-scheme:dark){:root{--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--b:#3987e5}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:30px 20px 70px}h1{font-size:28px;margin:0 0 6px}h2{font-size:20px;margin:40px 0 6px;padding-top:12px;border-top:1px solid var(--grid)}
h3{font-size:15px;margin:22px 0 4px}p{color:var(--ink2);margin:8px 0;max-width:82ch}.muted{color:var(--muted);font-size:13px}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{padding:6px 10px;border-bottom:1px solid var(--grid);text-align:left}
th{color:var(--ink2);font-size:12.5px}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.chart text{fill:var(--ink2);font-family:inherit}.chart .axis{font-size:11px;fill:var(--muted)}.chart .lab{font-size:12px;fill:var(--ink)}
.gal{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px}.slide{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:10px 12px}
.slide b{font-size:13px;word-break:break-all}.slide .pair{display:flex;gap:6px;margin:6px 0}.slide img{max-width:50%;height:auto;image-rendering:pixelated;border-radius:6px;background:#ecebe6}
.slide small{display:block;color:var(--muted);font-size:12px}code{font:12.5px ui-monospace,Menlo,monospace;background:var(--surface);border:1px solid var(--ring);border-radius:4px;padding:0 4px}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;color:var(--ink2);margin:8px 0}.legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px}
.note-box{border-left:3px solid var(--muted);padding:6px 14px;margin:14px 0;background:var(--surface);color:var(--ink2);border-radius:0 8px 8px 0}
"""
rgb = lambda c: "rgb(%d,%d,%d)" % COLOURS[c]
legend = "<div class='legend'>" + "".join(f"<span><i style='background:{rgb(c)}'></i>{e(c)}</span>" for c in CLASSES) + "</div>"


def bars_svg(labels, values, colours, width=560):
    rh, pl, pr = 26, 130, 60; H = rh * len(labels) + 8; vmax = max(max(values), 1e-9)
    out = [f'<svg class="chart" viewBox="0 0 {width} {H}" width="100%" role="img" aria-label="share of tiles by predicted class">']
    for i, (l, v, c) in enumerate(zip(labels, values, colours)):
        y = 4 + i * rh; w = (width - pl - pr) * v / vmax
        out.append(f'<text class="lab" x="{pl - 8}" y="{y + 14}" text-anchor="end">{e(l)}</text><rect x="{pl}" y="{y}" width="{max(w, 1):.1f}" height="16" rx="4" fill="{c}"/><text class="axis" x="{pl + w + 6:.1f}" y="{y + 13}">{v:.1%}</text>')
    return "".join(out) + "</svg>"


def hist_svg(values, bins=40, width=520, height=190, xlabel="mean confidence"):
    cnt, edges = np.histogram(values, bins=bins, range=(0.3, 1.0)); pl, pr, pt, pb = 44, 12, 8, 32
    sx = lambda z: pl + (z - 0.3) / 0.7 * (width - pl - pr); sy = lambda c: height - pb - c / max(cnt.max(), 1) * (height - pb - pt)
    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{e(xlabel)}">']
    for i, c in enumerate(cnt):
        if c: out.append(f'<rect x="{sx(edges[i]) + .5:.1f}" y="{sy(c):.1f}" width="{max(sx(edges[i + 1]) - sx(edges[i]) - 1, 1):.1f}" height="{height - pb - sy(c):.1f}" fill="var(--b)" rx="2"/>')
    for tk in (0.3, 0.5, 0.7, 0.9, 1.0): out.append(f'<text class="axis" x="{sx(tk):.1f}" y="{height - 14}" text-anchor="middle">{tk:g}</text>')
    return "".join(out) + f'<text class="axis" x="{width / 2}" y="{height - 1}" text-anchor="middle">{e(xlabel)}</text></svg>'


def shade(v, vmax):  # cell background for the composition table, one blue hue
    return f"background:rgba(42,120,214,{min(1, v / max(vmax, 1e-9)) * 0.55:.2f})"


comp = {t: np.array([np.average([r["share"][c] for r in ok.values() if r["type"] == t], weights=[r["tiles"] for r in ok.values() if r["type"] == t]) for c in CLASSES]) for t in types}
vmax = max(v.max() for v in comp.values())
tab = "".join(f"<tr><td>{e(t)}</td><td class='n'>{sum(r['type'] == t for r in ok.values())}</td><td class='n'>{sum(r['tiles'] for r in ok.values() if r['type'] == t):,}</td>"
              + "".join(f"<td class='n' style='{shade(v, vmax)}'>{v:.0%}</td>" for v in comp[t]) + "</tr>" for t in types)
conf_all = [r["mean_confidence"] for r in ok.values()]
def link(name):
    if not a.viewer_url:
        return e(name)
    return f"<a href='{e(a.viewer_url.format(slide=name), quote=True)}' target='_blank' rel='noopener' title='open in xOpat (needs the tunnel)'>{e(name)}</a>"


cards = "".join(
    f"<div class='slide'><b>{link(g['r']['slide'])}</b><div class='pair'><img alt='predicted class map' src='data:image/png;base64,{g['cls']}'><img alt='confidence map' src='data:image/png;base64,{g['conf']}'></div>"
    f"<small>type {e(g['r']['type'])} | {g['r']['tiles']:,} tiles | grid {g['grid'][0]} x {g['grid'][1]} | mean confidence {g['r']['mean_confidence']:.2f} | "
    + ", ".join(f"{c} {g['r']['share'][c]:.0%}" for c in sorted(CLASSES, key=lambda c: -g['r']['share'][c])[:3]) + "</small></div>" for g in gallery)
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Tissue classification of the Virchow2 tile features</title><style>{CSS}</style></head><body><main>
<h1>Tissue classification applied to the MMCI Virchow2 tile features</h1>
<p>The linear tissue classifier of MLflow experiment 104 (run <code>0e2230c7</code>) applied to a random subset of <b>{len(ok)}</b> slides ({total_tiles:,} tiles). Generated {time.strftime('%d %b %Y')}. The report holds class maps (one colour per 128 &micro;m tile), no slide pixels.</p>
<div class="note-box"><b>The model.</b> <code>Linear(2560 &rarr; 7)</code> and softmax on <code>[feats | average_patch_feats]</code> of the raw Virchow2 embeddings, trained on a mixed dataset (MOU, MUG, FNBrno, TCGA) with the classes Nerve, Blood, Connective-tissue, Fat, Epithelium, Muscle and Other. Each tile gets the probability of every class and the most likely class. <b>There is no lymphoid or lymphocyte class</b>, so lymph-node tissue is expected to be called Other, Blood or Connective-tissue; the maps show where fat, connective tissue and epithelium are, not the immune-cell content.</div>
{f"<p>{len(failed)} slides could not be processed: " + e(', '.join(r['slide'] for r in failed[:6])) + ".</p>" if failed else ""}
<h2>1. Class composition</h2>
{legend}
<div class="gal" style="grid-template-columns:repeat(auto-fit,minmax(360px,1fr))"><div><h3>Share of all tiles by predicted class</h3>{bars_svg(CLASSES, share_all.tolist(), [rgb(c) for c in CLASSES])}</div>
<div><h3>Mean confidence per slide (probability of the chosen class)</h3>{hist_svg(conf_all)}</div></div>
<h3>By slide type (name suffix), share of tiles</h3>
<div class="tw"><table><thead><tr><th>Type</th><th class="n">Slides</th><th class="n">Tiles</th>{''.join(f'<th class="n">{e(c)}</th>' for c in CLASSES)}</tr></thead><tbody>{tab}</tbody></table></div>
<p class="muted">Tile-weighted means over the slides of each type; darker cells = larger share.</p>
<h2>2. Class maps of individual slides</h2>
<p><b>Left:</b> predicted class of each tile. <b>Right:</b> confidence (probability of the chosen class) on a blue scale, dark = confident. Up to {a.per_type} slides per type, spread over the size range. Grey = no tile. <b>Click a slide name to open it in xOpat</b> (needs the viewer pod and the ssh tunnel); the tissue regions of these slides are served as <code>overlays/&lt;slide&gt;.tissue.json</code>.</p>
{legend}
<div class="gal">{cards}</div>
<p class="muted">Built in {time.time() - t0:.0f} s by reports/prediction_report.py.</p></main></body></html>"""
open(a.out, "w", encoding="utf-8").write(page)
print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")
