"""HTML report of the pre-extracted Virchow2 tile features (one .h5 per slide): inventory, feature statistics and
per-slide maps of the features over the tissue.

Format of a file (see the report): /data/feats (N, 1280) class token, /data/average_patch_feats (N, 1280) mean patch token,
/patch_grid/patch_coords (N, 2) float level-0 pixel coordinates of the tiles (256 px at 0.5 um/px, no overlap).
The tissue-classifier input is concat([feats, average_patch_feats]) (2560-d), which is also what the maps use.

Slides are only read; the report (embedded PNG / SVG, no slide pixels) is the only thing written.

usage: feature_report.py --features <dir> --out report.html [--slide-list names.txt] [--per-type 6] [--stats-slides 30]
"""
import argparse
import base64
import collections
import html
import io
import json
import os
import re
import time

import h5py
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--features", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--slide-list", default="", help="text file with the names of all slides, to find slides without features")
ap.add_argument("--per-type", type=int, default=6, help="slides per slide type in the map gallery")
ap.add_argument("--stats-slides", type=int, default=30, help="slides per type sampled for the feature statistics and the PCA")
ap.add_argument("--tiles-per-slide", type=int, default=1000); ap.add_argument("--map-size", type=int, default=300)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
e = html.escape
t0 = time.time()


def log(*s):
    print(f"[{time.time() - t0:5.0f}s]", *s, flush=True)


def slide_type(name):
    m = re.search(r"-([A-Za-z]+)_?$", name)
    return m.group(1) if m else "(none)"


# ── 1. inventory: metadata of every file ───────────────────────────────────────────────────────────────
files = sorted(f[:-3] for f in os.listdir(a.features) if f.endswith(".h5"))
inv = []
for name in files:
    try:
        with h5py.File(f"{a.features}/{name}.h5", "r") as h:
            n = h["data/feats"].shape[0]
            inv.append(dict(name=name, type=slide_type(name), n=n, dim=h["data/feats"].shape[1],
                            consistent=h["data/average_patch_feats"].shape[0] == n == h["patch_grid/patch_coords"].shape[0],
                            complete=str(h.attrs.get("extraction_complete", "")) == "True",
                            extractor=str(h.attrs.get("feature_extractor", ""))))
    except Exception as ex:  # unreadable file
        inv.append(dict(name=name, type=slide_type(name), n=0, dim=0, consistent=False, complete=False, extractor=f"ERROR {ex!r}"[:80]))
log("inventory", len(inv), "files")
missing = []
if a.slide_list:
    names = {l.strip().rsplit(".", 1)[0] for l in open(a.slide_list) if l.strip()}
    missing = sorted(names - set(files))
types = sorted({r["type"] for r in inv}, key=lambda t: -sum(r["type"] == t for r in inv))
by_type = {t: [r for r in inv if r["type"] == t] for t in types}

# ── 2. feature statistics and PCA on a sample of tiles ─────────────────────────────────────────────────
sample_X, sample_type, norms = [], [], {"feats": [], "avg": []}
for t in types:
    pool = [r for r in by_type[t] if r["n"] > 0]
    pick = [pool[i] for i in rng.permutation(len(pool))[: a.stats_slides]]
    for r in pick:
        with h5py.File(f"{a.features}/{r['name']}.h5", "r") as h:
            idx = np.sort(rng.choice(r["n"], min(r["n"], a.tiles_per_slide), replace=False))
            f, g = h["data/feats"][idx], h["data/average_patch_feats"][idx]
        sample_X.append(np.concatenate([f, g], 1)); sample_type += [t] * len(idx)
        norms["feats"].append(np.linalg.norm(f, axis=1)); norms["avg"].append(np.linalg.norm(g, axis=1))
X = np.concatenate(sample_X).astype(np.float32); sample_type = np.array(sample_type)
for k in norms:
    norms[k] = np.concatenate(norms[k])
mean = X.mean(0); Xc = X - mean
cov = (Xc.T @ Xc) / (len(Xc) - 1)
vals, vecs = np.linalg.eigh(cov.astype(np.float64)); order = np.argsort(vals)[::-1]; vals, vecs = vals[order], vecs[:, order]
evr = vals / vals.sum(); V = vecs[:, :3].astype(np.float32)
proj = Xc @ V
lo, hi = np.percentile(proj, 1, axis=0), np.percentile(proj, 99, axis=0)
log("PCA done on", len(X), "tiles; explained variance of PC1-3:", evr[:3].round(3).tolist())
frac_nan = float(np.isnan(X).mean())

# ── 3. per-slide maps ─────────────────────────────────────────────────────────────────────────────────
RAMP = np.array([[205, 226, 251], [134, 182, 239], [57, 135, 229], [28, 92, 171], [13, 54, 107]], dtype=np.float32)  # sequential blue


def ramp(v):  # v in 0..1 -> RGB
    p = np.clip(v, 0, 1) * (len(RAMP) - 1); i = np.minimum(p.astype(int), len(RAMP) - 2); w = (p - i)[:, None]
    return (RAMP[i] * (1 - w) + RAMP[i + 1] * w).astype(np.uint8)


def png_b64(arr):
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


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
    return np.asarray(im), (img.shape[1], img.shape[0]), step


gallery = []
for t in types:
    pool = sorted([r for r in by_type[t] if r["n"] >= 150], key=lambda r: r["n"])
    if not pool:
        continue
    picks = [pool[int(i)] for i in np.unique(np.linspace(0, len(pool) - 1, min(a.per_type, len(pool))).round())]
    for r in picks:
        with h5py.File(f"{a.features}/{r['name']}.h5", "r") as h:
            f, g, c = h["data/feats"][:], h["data/average_patch_feats"][:], h["patch_grid/patch_coords"][:]
        p = (np.concatenate([f, g], 1) - mean) @ V
        rgb = (np.clip((p - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        nrm = np.linalg.norm(f, axis=1); nrm_c = ramp((nrm - np.percentile(norms["feats"], 2)) / (np.percentile(norms["feats"], 98) - np.percentile(norms["feats"], 2)))
        m1, size, step = grid_map(c[:, 0], c[:, 1], rgb); m2, _, _ = grid_map(c[:, 0], c[:, 1], nrm_c)
        gallery.append(dict(r=r, pca=png_b64(m1), norm=png_b64(m2), grid=size, step=step, area=r["n"] * (step * 0.2428 / 1000) ** 2 if step else 0,
                            norm_mean=float(nrm.mean())))
    log("gallery", t, len(picks), "slides")

# ── 4. HTML ───────────────────────────────────────────────────────────────────────────────────────────
CSS = """
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--b:#2a78d6}
@media (prefers-color-scheme:dark){:root{--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--b:#3987e5}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:30px 20px 70px}h1{font-size:28px;margin:0 0 6px}h2{font-size:20px;margin:40px 0 6px;padding-top:12px;border-top:1px solid var(--grid)}
h3{font-size:15px;margin:22px 0 4px}p{color:var(--ink2);margin:8px 0;max-width:80ch}.muted{color:var(--muted);font-size:13px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:14px 16px;margin:12px 0}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{padding:6px 10px;border-bottom:1px solid var(--grid);text-align:left}
th{color:var(--ink2);font-size:12.5px}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.chart text{fill:var(--ink2);font-family:inherit}.chart .axis{font-size:11px;fill:var(--muted)}.chart .lab{font-size:12px;fill:var(--ink)}
.chart .grid{stroke:var(--grid)}.chart .base{stroke:var(--axis)}
.gal{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px}.slide{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:10px 12px}
.slide b{font-size:13px;word-break:break-all}.slide .pair{display:flex;gap:6px;margin:6px 0}.slide img{max-width:50%;height:auto;image-rendering:pixelated;border-radius:6px;background:#ecebe6}
.slide small{display:block;color:var(--muted);font-size:12px}code{font:12.5px ui-monospace,Menlo,monospace;background:var(--surface);border:1px solid var(--ring);border-radius:4px;padding:0 4px}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;border:1px solid var(--ring);font-size:12px;color:var(--ink2)}
.ok{color:#0c8a0c}.warn{color:#b45309}
"""


def hist_svg(values, bins, xlabel, log_x=False, width=520, height=200, color="var(--b)"):
    v = np.asarray(values, float); v = v[v > 0] if log_x else v
    edges = np.logspace(np.log10(v.min()), np.log10(v.max()), bins + 1) if log_x else np.linspace(v.min(), v.max(), bins + 1)
    cnt, _ = np.histogram(v, edges); pl, pr, pt, pb = 44, 12, 10, 34; W, H = width, height
    sx = (lambda z: pl + (np.log10(z) - np.log10(edges[0])) / (np.log10(edges[-1]) - np.log10(edges[0])) * (W - pl - pr)) if log_x else \
         (lambda z: pl + (z - edges[0]) / (edges[-1] - edges[0]) * (W - pl - pr))
    sy = lambda c: H - pb - c / max(cnt.max(), 1) * (H - pb - pt)
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{e(xlabel)}">']
    for k in range(4):
        y = H - pb - k / 3 * (H - pb - pt); out.append(f'<line class="grid" x1="{pl}" x2="{W - pr}" y1="{y:.1f}" y2="{y:.1f}"/><text class="axis" x="{pl - 6}" y="{y + 4:.1f}" text-anchor="end">{int(cnt.max() * k / 3)}</text>')
    for i, c in enumerate(cnt):
        x0, x1 = sx(edges[i]), sx(edges[i + 1])
        if c: out.append(f'<rect x="{x0 + 0.5:.1f}" y="{sy(c):.1f}" width="{max(x1 - x0 - 1, 1):.1f}" height="{H - pb - sy(c):.1f}" fill="{color}" rx="2"/>')
    ticks = [10 ** k for k in range(int(np.ceil(np.log10(edges[0]))), int(np.floor(np.log10(edges[-1]))) + 1)] if log_x else np.linspace(edges[0], edges[-1], 5)
    for tk in ticks:
        out.append(f'<text class="axis" x="{sx(tk):.1f}" y="{H - 14}" text-anchor="middle">{tk:,.0f}' + ('' if log_x else '') + '</text>')
    out.append(f'<text class="axis" x="{(pl + W - pr) / 2:.1f}" y="{H - 1}" text-anchor="middle">{e(xlabel)}</text><line class="base" x1="{pl}" x2="{W - pr}" y1="{H - pb}" y2="{H - pb}"/></svg>')
    return "".join(out)


def bars_svg(labels, values, xlabel, width=520, fmt="{:.1%}"):
    rh, pl, pr = 24, 70, 70; H = rh * len(labels) + 30; vmax = max(values)
    out = [f'<svg class="chart" viewBox="0 0 {width} {H}" width="100%" role="img" aria-label="{e(xlabel)}">']
    for i, (l, v) in enumerate(zip(labels, values)):
        y = 6 + i * rh; w = (width - pl - pr) * v / vmax
        out.append(f'<text class="lab" x="{pl - 8}" y="{y + 13}" text-anchor="end">{e(str(l))}</text><rect x="{pl}" y="{y}" width="{w:.1f}" height="14" rx="4" fill="var(--b)"/><text class="axis" x="{pl + w + 6:.1f}" y="{y + 12}">{fmt.format(v)}</text>')
    out.append(f'<text class="axis" x="{pl}" y="{H - 4}">{e(xlabel)}</text></svg>')
    return "".join(out)


def density_png(p, rng_x, rng_y, size=200):
    h, _, _ = np.histogram2d(p[:, 0], p[:, 1], bins=size // 2, range=[rng_x, rng_y])
    v = np.log1p(h.T[::-1]); v = v / max(v.max(), 1)
    img = np.where(v[..., None] > 0, ramp(v.ravel()).reshape(*v.shape, 3), np.array([236, 235, 230], np.uint8)).astype(np.uint8)
    return png_b64(np.asarray(Image.fromarray(img).resize((size, size), Image.NEAREST)))


total_tiles = sum(r["n"] for r in inv)
bad = [r for r in inv if not r["consistent"] or not r["complete"] or r["n"] == 0]
tab_rows = "".join(f"<tr><td>{e(t)}</td><td class='n'>{len(by_type[t]):,}</td><td class='n'>{sum(r['n'] for r in by_type[t]):,}</td><td class='n'>{int(np.median([r['n'] for r in by_type[t]])):,}</td>"
                   f"<td class='n'>{min(r['n'] for r in by_type[t]):,}</td><td class='n'>{max(r['n'] for r in by_type[t]):,}</td></tr>" for t in types)
dens = "".join(f"<div><div class='muted'>{e(t)} ({int((sample_type == t).sum()):,} tiles)</div><img alt='PC1 vs PC2 density, {e(t)}' style='width:100%;image-rendering:pixelated;border-radius:6px' "
               f"src='data:image/png;base64,{density_png(proj[sample_type == t][:, :2], (np.percentile(proj[:, 0], 0.5), np.percentile(proj[:, 0], 99.5)), (np.percentile(proj[:, 1], 0.5), np.percentile(proj[:, 1], 99.5)))}'></div>"
               for t in types if (sample_type == t).sum() > 50)
cards = "".join(
    f"<div class='slide'><b>{e(g['r']['name'])}</b><div class='pair'><img alt='PCA colour map' src='data:image/png;base64,{g['pca']}'><img alt='feature norm map' src='data:image/png;base64,{g['norm']}'></div>"
    f"<small>type {e(g['r']['type'])} | {g['r']['n']:,} tiles | grid {g['grid'][0]} x {g['grid'][1]} | tissue about {g['area']:.0f} mm&sup2; | mean |feats| {g['norm_mean']:.1f}</small></div>" for g in gallery)
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Virchow2 tile features of the MMCI slides</title><style>{CSS}</style></head><body><main>
<h1>Virchow2 tile features of the MMCI slides</h1>
<p>What is in the pre-extracted feature files, and what the features look like over the individual slides. Generated {time.strftime('%d %b %Y')} from <code>{e(a.features)}</code>. Slides were only read; this report holds feature maps (one colour per 128 &micro;m tile), no slide pixels.</p>
<h2>1. File format</h2>
<div class="card tw"><table><thead><tr><th>Item</th><th>Content</th></tr></thead><tbody>
<tr><td><code>/data/feats</code></td><td>(N &times; 1280) float32: Virchow2 class-token embedding of every tile</td></tr>
<tr><td><code>/data/average_patch_feats</code></td><td>(N &times; 1280) float32: mean of the patch tokens of every tile</td></tr>
<tr><td><code>/patch_grid/patch_coords</code></td><td>(N &times; 2) float64: level-0 pixel position (x, y) of each tile; regular grid, step about 527 px = 128 &micro;m</td></tr>
<tr><td><code>/mask/mask_pickle</code></td><td>pickled tissue mask (not loaded: unpickling can run code)</td></tr>
<tr><td>attributes</td><td><code>extraction_complete</code>, <code>extraction_identity</code> (mask configuration), <code>feature_extractor</code> (Virchow2Model), <code>feature_shapes</code>, <code>wsi_id</code>; per-slide <code>feats_mean/std</code> and <code>average_patch_feats_mean/std</code> (1280 each)</td></tr>
<tr><td>tiling</td><td>256 px at 0.5 &micro;m/px, no overlap, every tile that intersects the tissue mask (whole slide, not only lymph nodes)</td></tr>
<tr><td>model input</td><td>the tissue classifier expects <code>concat([feats, average_patch_feats])</code> (2560-d)</td></tr></tbody></table></div>
<h2>2. Inventory</h2>
<p><b>{len(inv):,}</b> files, <b>{total_tiles:,}</b> tiles. {'All files complete and internally consistent.' if not bad else f'<span class="warn">{len(bad)} files are incomplete, empty or inconsistent: ' + e(', '.join(r['name'] for r in bad[:8])) + '</span>'} Feature dimension: {sorted({r['dim'] for r in inv if r['dim']})}. Extractor: {e(', '.join(sorted({r['extractor'] for r in inv})))}. Non-finite values in the sampled tiles: {frac_nan:.2%}.
{('Slides without a feature file: ' + e(', '.join(missing)) + '.') if a.slide_list else ''}</p>
<div class="tw"><table><thead><tr><th>Slide type (name suffix)</th><th class="n">Slides</th><th class="n">Tiles</th><th class="n">Median tiles</th><th class="n">Min</th><th class="n">Max</th></tr></thead><tbody>{tab_rows}</tbody></table></div>
<div class="gal" style="grid-template-columns:repeat(auto-fit,minmax(320px,1fr))"><div><h3>Tiles per slide</h3>{hist_svg([r['n'] for r in inv], 40, 'tiles per slide (log scale)', log_x=True)}</div>
<div><h3>Tiles by slide type</h3>{bars_svg(types, [sum(r['n'] for r in by_type[t]) / total_tiles for t in types], 'share of all tiles')}</div></div>
<h2>3. The features</h2>
<p>Sampled {len(X):,} tiles ({a.tiles_per_slide:,} per slide, up to {a.stats_slides} slides per type). The two halves have different scales: the mean norm is {norms['feats'].mean():.1f} for the class token and {norms['avg'].mean():.1f} for the mean patch token.</p>
<div class="gal" style="grid-template-columns:repeat(auto-fit,minmax(320px,1fr))"><div><h3>Norm of the class token</h3>{hist_svg(norms['feats'], 40, 'norm of feats')}</div>
<div><h3>Norm of the mean patch token</h3>{hist_svg(norms['avg'], 40, 'norm of average_patch_feats', color='var(--b)')}</div>
<div><h3>Variance explained by PC 1-10</h3>{bars_svg([f'PC{i + 1}' for i in range(10)], evr[:10].tolist(), 'share of the total variance')}</div></div>
<p>PCA over the sampled tiles (2560-d, centred): the first three components explain {evr[:3].sum():.0%} of the variance (PC1 {evr[0]:.0%}, PC2 {evr[1]:.0%}, PC3 {evr[2]:.0%}). They colour the maps below.</p>
<h3>Tile distribution along PC1 and PC2, by slide type</h3><div class="gal" style="grid-template-columns:repeat(auto-fill,minmax(150px,1fr))">{dens}</div>
<h2>4. Features over individual slides</h2>
<p>Each card shows two maps of the same slide, one pixel per tile. <b>Left:</b> PC1-3 of the 2560-d feature as red, green and blue, so tissue with similar features has a similar colour (colours are comparable between slides, the PCA is shared). <b>Right:</b> the norm of the class token on a blue scale (dark = high). Slides are spread over the size range of each type ({a.per_type} per type). Grey = no tile (background).</p>
<div class="gal">{cards}</div>
<p class="muted">Built in {time.time() - t0:.0f} s by reports/feature_report.py.</p></main></body></html>"""
open(a.out, "w", encoding="utf-8").write(page)
log("wrote", a.out, f"{os.path.getsize(a.out) / 1e6:.1f} MB")
