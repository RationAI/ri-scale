"""Builds reports/survival_report.html: a self-contained report (inline SVG, no external resources) of the survival work.

Aggregate numbers only, copied from the logged runs (MLflow experiment 72, results folders on musica). No patient-level data.
usage: python reports/build_report.py
"""
import html
import math
from pathlib import Path

OUT = Path(__file__).with_name("survival_report.html")

# ── data (all aggregate; see the run ids in the appendix) ─────────────────────────────────────────────

TAB_VARIANTS = [  # label, family, cv mean, cv sd, test, test lo, test hi, attributes note
    ("Cox, all features", "tab", 0.749, 0.024, 0.742, 0.617, 0.865),
    ("Cox, stage group only", "tab", 0.673, 0.041, 0.661, 0.508, 0.800),
    ("Elastic-net Cox (coxnet)", "tab", 0.758, 0.041, None, None, None),
    ("Cox, min_category_count = 5", "tab", 0.756, 0.052, None, None, None),
    ("Cox, cancer-specific survival", "tab", 0.749, 0.029, None, None, None),
    ("Stage only, cancer-specific", "tab", 0.693, 0.045, None, None, None),
]

HR = [  # feature level, HR, lo, hi, group
    ("Age (per year)", 1.047, 1.015, 1.080), ("Nodes examined (per node)", 0.960, 0.917, 1.004),
    ("Sex: female", 0.822, 0.464, 1.455), ("Location: rectum", 1.368, 0.653, 2.865), ("Location: right colon", 0.649, 0.318, 1.328),
    ("Grade: high", 1.047, 0.537, 2.043), ("pT3", 3.105, 1.434, 6.724), ("pT4", 1.865, 0.542, 6.419),
    ("pN1", 0.899, 0.484, 1.668), ("pN2", 1.633, 0.691, 3.858), ("M1", 4.709, 2.349, 9.440),
    ("Neoadjuvant therapy", 0.392, 0.131, 1.171), ("MMR not tested", 2.756, 1.393, 5.450), ("KRAS mutated", 6.182, 2.510, 15.225),
    ("KRAS not tested", 3.702, 0.631, 21.731), ("NRAS not tested", 0.441, 0.101, 1.931), ("BRAF not tested", 0.345, 0.108, 1.106),
]

DROP_ONE = [  # feature, delta C-index, holm p, folds improved
    ("M (metastasis)", 0.0415, 0.0002, "4/5"), ("KRAS", 0.0371, 0.0036, "5/5"), ("Age", 0.0317, 0.0265, "4/5"),
    ("pT", 0.0147, 0.0664, "3/5"), ("MMR", 0.0092, 0.0247, "3/5"), ("Neoadjuvant", 0.0042, 0.5306, "2/5"),
    ("Nodes examined", 0.0013, 0.5148, "3/5"), ("Sex", -0.0002, 1.0, "3/5"), ("Location", -0.0010, 0.6707, "2/5"),
    ("BRAF", -0.0028, 0.5394, "3/5"), ("NRAS", -0.0051, 1.0, "2/5"), ("pN", -0.0079, 1.0, "3/5"), ("Grade", -0.0097, 1.0, "0/5"),
]

MUG_VARIANTS = [  # label, n, events, cv c, sd, auc5
    ("All years, with stage IV", 3392, 2305, 0.641, 0.016, 0.675),
    ("All years, without stage IV", 3374, 2294, 0.641, 0.017, 0.677),
    ("From 2000, with stage IV", 1534, 935, 0.682, 0.013, 0.729),
    ("From 2000, without stage IV", 1526, 930, 0.682, 0.013, 0.728),
]
MUG_ERA = [("1980s", 896, 0.512), ("1990s", 1423, 0.571), ("2000s", 1590, 0.603), ("2010-13", 331, 0.535)]

IMG = [  # key, label, family, c, sd, auc5, gain
    ("clinical", "Age + sex only", "base", 0.623, 0.064, 0.620, 0.0),
    ("pooled_pca8", "Pooled node embedding (PCA 8)", "img", 0.627, 0.060, 0.639, 0.005),
    ("pooled_pca8_hand", "Pooled + node count / area", "img", 0.623, 0.063, 0.644, 0.001),
    ("pooled_pca16", "Pooled node embedding (PCA 16)", "img", 0.606, 0.066, 0.614, -0.017),
    ("mil", "Attention over nodes + age, sex", "img", 0.616, 0.047, 0.630, -0.007),
    ("mil_image_only", "Attention over nodes, images only", "img", 0.613, 0.044, 0.628, -0.009),
    ("ratio", "Node ratio (detector)", "ratio", 0.656, 0.066, 0.678, 0.033),
    ("ratio_full", "Node ratio + positives + max + count", "ratio", 0.670, 0.061, 0.693, 0.048),
    ("burden", "Positive nodes + max + count", "ratio", 0.674, 0.058, 0.699, 0.051),
    ("tile_ratio", "Tile-level: ratio + top-tile score", "ratio", 0.666, 0.070, 0.697, 0.044),
    ("tile_burden", "Tile-level: positives + area + score", "ratio", 0.663, 0.064, 0.699, 0.040),
    ("tile_full", "Tile-level: all tile features", "ratio", 0.660, 0.061, 0.696, 0.037),
    ("pooled_pca8_tnm", "Pooled embedding + pT/pN/M", "img", 0.721, 0.069, 0.772, 0.098),
    ("excel_tnm", "Excel: age, sex, pT, pN, M", "tab", 0.740, 0.055, 0.787, 0.118),
    ("excel_full", "Excel: full feature set", "tab", 0.753, 0.049, 0.788, 0.131),
]

STAGING_AUC = [("pN positive", 0.89, 0.92), ("pT3/T4", 0.77, 0.81), ("M1", 0.69, 0.72), ("Female", 0.73, 0.69), ("Age 70+", 0.65, 0.65)]

SEG_SIZE = [("< 3,000", 15, 0.0, 0.200), ("3,000-6,000", 27, 0.333, 0.481), ("6,000-12,000", 60, 0.517, 0.850),
            ("12,000-30,000", 197, 0.878, 0.949), ("30,000+", 431, 0.944, 0.961)]

# ── model x attribute matrix ──────────────────────────────────────────────────────────────────────────
# "c" = clinical input, "i" = derived from the images, "l" = training label only (never an input), "" = not used
ATTRS = [  # key, header, group
    ("age", "Age", "Clinical / pathology data"), ("sex", "Sex", "Clinical / pathology data"),
    ("nodes", "Nodes examined", "Clinical / pathology data"), ("loc", "Location", "Clinical / pathology data"),
    ("morph", "Morphology", "Clinical / pathology data"), ("grade", "Grade", "Clinical / pathology data"),
    ("pt", "pT", "Clinical / pathology data"), ("pn", "pN", "Clinical / pathology data"), ("m", "M", "Clinical / pathology data"),
    ("neo", "Neo-adjuvant", "Clinical / pathology data"), ("stage", "Stage group", "Clinical / pathology data"),
    ("mmr", "MMR", "Clinical / pathology data"), ("kras", "KRAS", "Clinical / pathology data"),
    ("nras", "NRAS", "Clinical / pathology data"), ("braf", "BRAF", "Clinical / pathology data"),
    ("emb", "Node embeddings", "From the lymph-node images"), ("hand", "Node count / area", "From the lymph-node images"),
    ("ratio", "Node ratio / burden", "From the lymph-node images"), ("tile", "Tile-level burden", "From the lymph-node images"),
    ("lab", "Positive-node count", "Training label only"),
]
DEFAULT = dict.fromkeys(["age", "nodes", "sex", "loc", "morph", "grade", "pt", "pn", "m", "neo", "mmr", "kras", "nras", "braf"], "c")
MATRIX = [  # section, model, family, marks, note
    ("MMCI, Excel sheet (220 patients)", "Cox, default features", "tab", DEFAULT, "Morphology is constant in this cohort and is dropped by the encoder."),
    ("", "Cox, stage group only", "tab", {"stage": "c"}, "Clinical reference."),
    ("", "Elastic-net Cox", "tab", DEFAULT, "Same inputs, penalised."),
    ("", "Cox, min_category_count = 5", "tab", DEFAULT, "Rarer levels (e.g. dMMR) kept."),
    ("", "Cancer-specific survival", "tab", DEFAULT, "Same inputs, different endpoint."),
    ("MUG cohort (4,240 patients)", "Shared-feature Cox", "tab", {"age": "c", "sex": "c", "grade": "c", "pt": "c", "pn": "c", "m": "c"},
     "M is constant when stage IV is excluded and is dropped."),
    ("Image study, MMCI (214 patients, 87 deaths)", "Age + sex only", "base", {"age": "c", "sex": "c"}, "The floor."),
    ("", "Excel reference: TNM", "tab", {"age": "c", "sex": "c", "pt": "c", "pn": "c", "m": "c"}, "Staging the images are meant to replace."),
    ("", "Excel reference: full", "tab", DEFAULT, "As the first model, on the same 214 patients."),
    ("", "Pooled node embedding", "img", {"age": "c", "sex": "c", "emb": "i"}, "Average of the node embeddings, PCA 8 / 16."),
    ("", "Pooled + node count / area", "img", {"age": "c", "sex": "c", "emb": "i", "hand": "i"}, "Adds log node count and log total / mean / max area."),
    ("", "Attention over nodes", "img", {"age": "c", "sex": "c", "emb": "i"}, "Also run without age and sex (images only)."),
    ("", "Pooled embedding + TNM", "img", {"age": "c", "sex": "c", "pt": "c", "pn": "c", "m": "c", "emb": "i"}, "Do images add to staging?"),
    ("", "Node ratio / burden", "ratio", {"age": "c", "sex": "c", "ratio": "i", "lab": "l"},
     "Detector scores each node; trained only with 'patient has a positive node'. The count is never an input."),
    ("", "Tile-level burden", "ratio", {"age": "c", "sex": "c", "tile": "i", "lab": "l"}, "Tile detector (PCA 128, top-64 tiles), trained with the same weak label."),
]

GLOSSARY = [  # attribute, source, derivation, meaning
    ("Age", "birth date, diagnosis date", "(diagnosis - birth) / 365.25; outside 15-110 treated as missing", "Age at diagnosis in years."),
    ("Sex", "pohlavi", "M / F (codes 1 / 2 accepted)", "Recorded sex."),
    ("Nodes examined", "regional + sentinel node counts", "sum of both counts", "Lymph nodes examined by pathology; a quality marker."),
    ("Location", "topografie (else diagnoza)", "ICD-10: right C18.0-4, left C18.5-7, rectum C19-20; C18.8/9 unknown", "Tumour site."),
    ("Morphology", "morfologie", "ICD-O-3: adenocarcinoma, mucinous (8480/1), signet ring (8490), other", "Histological type. 221 of 225 records are adenocarcinoma."),
    ("Grade", "grading", "G1-G2 = low, G3-G4 = high", "Differentiation grade."),
    ("pT", "pt", "T0-2 / T3 / T4 (substages merged)", "Depth of tumour invasion."),
    ("pN", "pn (else node counts)", "N0 / N1 / N2 (N3 only in old MUG data)", "Regional node involvement."),
    ("M", "pm, cm", "M0 / M1; pathological M wins unless only the clinical one shows metastases", "Distant metastasis."),
    ("Neo-adjuvant", "y_prefix or 'y' in pT/pN", "yes / no", "Treated before surgery."),
    ("Stage group", "klinicke_stadium (else TNM)", "0-I / II / III / IV", "Only used in the stage-only reference model."),
    ("MMR", "mmr", "pMMR / dMMR / not tested (dMMR = 10 patients, merged into the reference)", "Mismatch-repair status."),
    ("KRAS, NRAS, BRAF", "kras, nras, braf", "wt / mutated / not tested (mutated is rare for NRAS and BRAF)", "Molecular markers; mostly tested in metastatic patients."),
    ("Node embeddings", "SAM3 node masks + Virchow2", "mean of the 2,560-d tile embeddings inside each node (class token | mean patch token)", "What a node looks like."),
    ("Node count / area", "SAM3 node masks", "log node count; log total / mean / max node area (mm2)", "Size of the nodal tissue."),
    ("Node ratio / burden", "node embeddings", "detector -> expected positive nodes, share of nodes, highest score", "Image-derived lymph-node ratio and nodal tumour burden."),
    ("Tile-level burden", "tile embeddings (PCA 128)", "mean of the top-64 tile scores; nodes with 8+ tumour tiles count as positive", "Tumour inside the nodes at tile resolution."),
    ("Positive-node count", "node counts in the sheet", "positive nodes > 0", "Training label of the detector only; never a model input."),
]

# ── SVG helpers ───────────────────────────────────────────────────────────────────────────────────────

FAM = {"tab": "var(--c-tab)", "img": "var(--c-img)", "ratio": "var(--c-ratio)", "base": "var(--c-base)"}
FAM_NAME = {"tab": "Clinical / staging data", "img": "Image-derived", "ratio": "Image-derived node ratio", "base": "Age + sex only"}
e = html.escape


def tip(*parts):
    return e(" | ".join(parts), quote=True)


def svg_open(w, h, label):
    return f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{e(label)}" class="chart">'


def xscale(lo, hi, x0, x1, log=False):
    if log:
        return lambda v: x0 + (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)) * (x1 - x0)
    return lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)


def hbar_path(x0, x1, y, h, r=4):
    w = x1 - x0
    if w < 1:
        return ""
    r = min(r, w, h / 2)
    return f"M{x0:.1f},{y:.1f} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{h - 2 * r:.1f} a{r},{r} 0 0 1 {-r},{r} h{-(w - r):.1f} z"


def vbar_path(x, y0, w, h, r=4):
    """Column growing up from baseline y0, rounded data end (top)."""
    if h < 1:
        return ""
    r = min(r, w / 2, h)
    return f"M{x:.1f},{y0:.1f} v{-(h - r):.1f} a{r},{r} 0 0 1 {r},{-r} h{w - 2 * r:.1f} a{r},{r} 0 0 1 {r},{r} v{h - r:.1f} z"


def dot_chart(rows, lo, hi, ticks, ref=None, ref_label="", chance=None, fmt="{:.3f}", width=820, label_w=300, row_h=30, title="", log=False, group_gap=30):
    """Dot + whisker chart. rows: (label, family, value, err_lo, err_hi, tip, group_header_or_None)."""
    pad_t, pad_b = 26, 30
    ngroups = sum(1 for r in rows if r[6])
    h = pad_t + row_h * len(rows) + group_gap * ngroups + pad_b
    x0, x1 = label_w, width - 50
    sx = xscale(lo, hi, x0, x1, log)
    out = [svg_open(width, h, title)]
    for t in ticks:
        x = sx(t)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{pad_t - 6}" y2="{h - pad_b + 2}" class="grid"/>'
                   f'<text x="{x:.1f}" y="{h - 8}" text-anchor="middle" class="axis">{t:g}</text>')
    if chance is not None:
        x = sx(chance)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{pad_t - 6}" y2="{h - pad_b + 2}" class="ref"/>'
                   f'<text x="{x + 4:.1f}" y="{pad_t - 10}" class="note">chance</text>')
    if ref is not None:
        x = sx(ref)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{pad_t - 6}" y2="{h - pad_b + 2}" class="ref2"/>'
                   f'<text x="{x + 4:.1f}" y="{pad_t - 10}" class="note">{e(ref_label)}</text>')
    y = pad_t
    for label, fam, v, a, b, tp, grp in rows:
        if grp:
            out.append(f'<text x="0" y="{y + 16}" class="grp">{e(grp)}</text>')
            y += group_gap
        cy = y + row_h / 2
        out.append(f'<text x="{x0 - 12}" y="{cy + 4:.1f}" text-anchor="end" class="lab">{e(label)}</text>')
        if a is not None:
            out.append(f'<line x1="{sx(a):.1f}" x2="{sx(b):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" class="whisk"/>'
                       f'<line x1="{sx(a):.1f}" x2="{sx(a):.1f}" y1="{cy - 4:.1f}" y2="{cy + 4:.1f}" class="whisk"/>'
                       f'<line x1="{sx(b):.1f}" x2="{sx(b):.1f}" y1="{cy - 4:.1f}" y2="{cy + 4:.1f}" class="whisk"/>')
        out.append(f'<circle cx="{sx(v):.1f}" cy="{cy:.1f}" r="5" fill="{FAM[fam]}" class="dot hit" data-tip="{tip(label, tp)}"/>')
        out.append(f'<rect x="{x0 - 320}" y="{y:.1f}" width="{x1 - x0 + 340}" height="{row_h}" fill="transparent" class="hit" data-tip="{tip(label, tp)}"/>')
        y += row_h
    out.append(f'<line x1="{x0}" x2="{x1}" y1="{h - pad_b + 2}" y2="{h - pad_b + 2}" class="base"/></svg>')
    return "".join(out)


def bar_rows(rows, vmax, ticks, fmt="{:.3f}", width=820, label_w=260, row_h=30, title="", pct=False, signed=False, tick_fmt=None):
    """Horizontal bars from a zero baseline. rows: (label, family, value, tip, marker_text_or_None)."""
    pad_t, pad_b = 12, 30
    h = pad_t + row_h * len(rows) + pad_b
    x0, x1 = label_w, width - 90
    lo = min(0.0, min(r[2] for r in rows)) if signed else 0.0
    sx = xscale(lo, vmax, x0, x1)
    out = [svg_open(width, h, title)]
    for t in ticks:
        x = sx(t)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{pad_t - 4}" y2="{h - pad_b + 2}" class="grid"/>'
                   f'<text x="{x:.1f}" y="{h - 8}" text-anchor="middle" class="axis">{tick_fmt(t) if tick_fmt else f"{t:g}"}</text>')
    zero = sx(0)
    for i, (label, fam, v, tp, mark) in enumerate(rows):
        y = pad_t + i * row_h + (row_h - 16) / 2
        out.append(f'<text x="{x0 - 12}" y="{y + 12:.1f}" text-anchor="end" class="lab">{e(label)}</text>')
        if v >= 0:
            out.append(f'<path d="{hbar_path(zero, sx(v), y, 16)}" fill="{FAM[fam]}" class="hit" data-tip="{tip(label, tp)}"/>')
            tx, anchor = sx(v) + 8, "start"
        else:
            out.append(f'<path d="{hbar_path(sx(v), zero, y, 16)}" fill="{FAM[fam]}" class="hit" data-tip="{tip(label, tp)}"/>')
            tx, anchor = zero + 8, "start"
        out.append(f'<text x="{tx:.1f}" y="{y + 12:.1f}" text-anchor="{anchor}" class="val">{fmt.format(v)}{(" " + e(mark)) if mark else ""}</text>')
    out.append(f'<line x1="{zero:.1f}" x2="{zero:.1f}" y1="{pad_t - 4}" y2="{h - pad_b + 2}" class="base"/></svg>')
    return "".join(out)


def grouped_columns(cats, series, width=820, height=320, title="", ymax=1.0, pct=True, what="of nodes found"):
    """cats: (label, n or None, v1, v2); series: two (name, family). Columns from the zero baseline, values 0..1."""
    pad_l, pad_r, pad_t, pad_b = 46, 20, 24, 58
    x0, x1, y0, y1 = pad_l, width - pad_r, height - pad_b, pad_t
    out = [svg_open(width, height, title)]
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        y = y0 - (y0 - y1) * t / ymax
        out.append(f'<line x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/><text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end" class="axis">{(str(int(t * 100)) + "%") if pct else f"{t:g}"}</text>')
    band = (x1 - x0) / len(cats); bw = 20
    for i, (label, n, v1, v2) in enumerate(cats):
        cx = x0 + band * (i + 0.5)
        for k, (v, (name, fam)) in enumerate(zip((v1, v2), series)):
            x = cx - bw - 1 + k * (bw + 2)
            hh = (y0 - y1) * v / ymax
            out.append(f'<path d="{vbar_path(x, y0, bw, hh)}" fill="{FAM[fam]}" class="hit" data-tip="{tip(label + (f" (n = {n})" if n else ""), f"{name}: " + (f"{v * 100:.1f}% {what}" if pct else f"AUC {v:.2f}"))}"/>')
            out.append(f'<text x="{x + bw / 2:.1f}" y="{y0 - hh - 6:.1f}" text-anchor="middle" class="val">{(f"{v * 100:.0f}") if pct else f"{v:.2f}"}</text>')
        out.append(f'<text x="{cx:.1f}" y="{y0 + 18}" text-anchor="middle" class="lab">{e(label)}</text>'
                   + (f'<text x="{cx:.1f}" y="{y0 + 34}" text-anchor="middle" class="note">n = {n}</text>' if n else ""))
    out.append(f'<line x1="{x0}" x2="{x1}" y1="{y0}" y2="{y0}" class="base"/></svg>')
    return "".join(out)


def legend(items):
    return '<div class="legend">' + "".join(f'<span><i style="background:{FAM[f]}"></i>{e(n)}</span>' for f, n in items) + "</div>"


def table(headers, rows, cls=""):
    th = "".join(f"<th>{e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def view_table(headers, rows):
    return f'<details class="tv"><summary>Table view</summary>{table(headers, rows)}</details>'


# ── charts ────────────────────────────────────────────────────────────────────────────────────────────

def chart_tab_variants():
    rows = [(l, f, c, c - s, c + s, f"CV C-index {c:.3f} +/- {s:.3f}" + (f" | held-out test {t:.3f} ({lo:.2f}-{hi:.2f})" if t else ""), None)
            for l, f, c, s, t, lo, hi in TAB_VARIANTS]
    return dot_chart(rows, 0.55, 0.85, [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85], title="Cross-validated C-index of the MMCI tabular models")


def chart_forest():
    rows = [(l, "tab", hr, lo, hi, f"HR {hr:.2f} (95% CI {lo:.2f}-{hi:.2f})", None) for l, hr, lo, hi in HR]
    return dot_chart(rows, 0.1, 25, [0.1, 0.25, 0.5, 1, 2, 5, 10, 25], ref=1, ref_label="no effect", title="Adjusted hazard ratios, Cox model on all features",
                     log=True, width=820, label_w=240, row_h=26)


def chart_drop_one():
    rows = [(l, "tab" if p < 0.05 else "base", d, f"drop-one change in CV C-index {d:+.4f} | Holm-adjusted p {p:.3g} | better in {fi} folds", "*" if p < 0.05 else None)
            for l, d, p, fi in DROP_ONE]
    return bar_rows(rows, 0.05, [0, 0.01, 0.02, 0.03, 0.04, 0.05], fmt="{:+.3f}", title="Loss of C-index when a feature is removed", signed=True)


def chart_mug():
    rows = [(l, "tab", c, c - s, c + s, f"{n:,} patients, {ev:,} deaths | CV C-index {c:.3f} +/- {s:.3f} | AUC at 5 years {a:.3f}", None)
            for l, n, ev, c, s, a in MUG_VARIANTS]
    return dot_chart(rows, 0.60, 0.72, [0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72], title="MUG shared-feature Cox, four cohort choices", label_w=260)


def chart_era():
    rows = [(l, "tab", v, f"{n:,} patients diagnosed in the {l} | 5-year overall survival {v * 100:.0f}%", None) for l, n, v in MUG_ERA]
    return bar_rows([(l, f, v, t, None) for l, f, v, t, _ in rows], 1.0, [0, 0.25, 0.5, 0.75, 1.0], fmt="{:.0%}", title="MUG: 5-year overall survival by decade of diagnosis", label_w=120, tick_fmt=lambda v: f"{v:.0%}")


def chart_image():
    rows = []
    last = None
    for key, label, fam, c, s, a, g in IMG:
        grp = {"base": "Floor", "img": "Image models", "ratio": "Image models, lymph-node ratio", "tab": "Reference from the Excel sheet"}[fam]
        if key == "pooled_pca8_tnm":
            grp = "Images added to staging"
        show = grp if grp != last else None
        last = grp
        rows.append((label, fam, c, c - s, c + s, f"CV C-index {c:.3f} +/- {s:.3f} | AUC at 5 years {a:.3f} | gain over age + sex {g:+.3f}", show))
    return dot_chart(rows, 0.5, 0.85, [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85], ref=0.623, ref_label="age + sex", title="Image-based survival models, 5 x 5 repeated cross-validation",
                     label_w=300, row_h=26)


def chart_staging():
    return grouped_columns([(l, None, a, b) for l, a, b in STAGING_AUC], [("mean pooling", "img"), ("max pooling", "ratio")],
                           title="How well pooled node embeddings predict staging labels", ymax=1.0, pct=False)


def chart_seg():
    return grouped_columns([(l, n, o, nw) for l, n, o, nw in SEG_SIZE], [("MMCI-only decoder", "img"), ("MMCI + MUG decoder", "ratio")], title="Nodes found by node size on 135 held-out MUG slides")


# ── page ──────────────────────────────────────────────────────────────────────────────────────────────

def matrix_html():
    groups = []
    for key, head, grp in ATTRS:
        if not groups or groups[-1][0] != grp:
            groups.append([grp, 0])
        groups[-1][1] += 1
    th1 = '<th rowspan="2" class="mhead">Model</th>' + "".join(f'<th colspan="{n}" class="gh">{e(g)}</th>' for g, n in groups)
    th2 = "".join(f'<th class="vh"><span>{e(h)}</span></th>' for _, h, _ in ATTRS)
    sym = {"c": '<span class="mk mk-c" title="clinical input">&#9679;</span>', "i": '<span class="mk mk-i" title="derived from images">&#9670;</span>',
           "l": '<span class="mk mk-l" title="training label only">&#9675;</span>', "": ""}
    body = []
    for sec, model, fam, marks, note in MATRIX:
        if sec:
            body.append(f'<tr class="sec"><td colspan="{len(ATTRS) + 1}">{e(sec)}</td></tr>')
        cells = "".join(f'<td>{sym[marks.get(k, "")]}</td>' for k, _, _ in ATTRS)
        body.append(f'<tr><th class="rowh"><i style="background:{FAM[fam]}"></i>{e(model)}<small>{e(note)}</small></th>{cells}</tr>')
    return f'<div class="tw"><table class="matrix"><thead><tr>{th1}</tr><tr>{th2}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


CSS = """
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);
--c-tab:#2a78d6;--c-img:#eb6834;--c-ratio:#1baf7a;--c-base:#898781;--good:#006300;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
--c-tab:#3987e5;--c-img:#d95926;--c-ratio:#199e70;--c-base:#898781;--good:#0ca30c;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
--c-tab:#3987e5;--c-img:#d95926;--c-ratio:#199e70;--c-base:#898781;--good:#0ca30c;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1040px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:30px;margin:0 0 6px;letter-spacing:-.01em}h2{font-size:21px;margin:44px 0 6px;padding-top:12px;border-top:1px solid var(--grid)}h3{font-size:16px;margin:26px 0 4px}
p{margin:8px 0;color:var(--ink2);max-width:78ch}.sub{color:var(--ink2);margin:0 0 4px}.aud{font-size:13px;color:var(--muted)}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:16px 18px;margin:14px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:18px 0}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:14px 16px}.tile b{display:block;font-size:30px;font-weight:600;letter-spacing:-.01em}
.tile span{display:block;color:var(--ink2);font-size:13px}.tile small{display:block;color:var(--muted);font-size:12px;margin-top:4px}
.chart{display:block;max-width:100%;overflow:visible}.chart text{font-family:inherit;fill:var(--ink2)}
.chart .lab{font-size:12.5px;fill:var(--ink)}.chart .axis,.chart .note{font-size:11px;fill:var(--muted)}.chart .val{font-size:11.5px;fill:var(--ink2)}
.chart .grp{font-size:11px;fill:var(--muted);letter-spacing:.06em;text-transform:uppercase}
.chart .grid{stroke:var(--grid);stroke-width:1}.chart .base{stroke:var(--axis);stroke-width:1}.chart .ref{stroke:var(--axis);stroke-width:1}
.chart .ref2{stroke:var(--ink2);stroke-width:1;opacity:.55}.chart .whisk{stroke:var(--muted);stroke-width:1.5}
.chart .dot{stroke:var(--surface);stroke-width:2}.hit{cursor:default}.hit:hover{opacity:.85}
.legend{display:flex;flex-wrap:wrap;gap:16px;margin:8px 0 2px;font-size:13px;color:var(--ink2)}.legend i,.rowh i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:7px}
.tw{overflow-x:auto;margin:10px 0}table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{padding:7px 10px;border-bottom:1px solid var(--grid);text-align:left;vertical-align:top}
th{color:var(--ink2);font-weight:600;font-size:12.5px}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.matrix{font-size:12.5px;min-width:980px}.matrix th.gh{text-align:center;border-left:1px solid var(--grid);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
.matrix th.vh{height:112px;padding:0 2px;vertical-align:bottom;text-align:center;border-left:1px solid var(--grid)}
.matrix th.vh span{display:inline-block;writing-mode:vertical-rl;transform:rotate(180deg);white-space:nowrap;font-weight:500;padding-bottom:6px}
.matrix td{text-align:center;padding:5px 2px;border-left:1px solid var(--grid);width:34px}.matrix .mhead{vertical-align:bottom;min-width:270px}
.matrix .rowh{font-weight:500;color:var(--ink);min-width:270px}.matrix .rowh small{display:block;font-weight:400;color:var(--muted);font-size:11.5px;margin:1px 0 0 17px}
.matrix tr.sec td{background:var(--surface);color:var(--ink2);font-weight:600;font-size:12px;text-align:left;border-left:0}
.mk{font-size:15px;line-height:1}.mk-c{color:var(--c-tab)}.mk-i{color:var(--c-ratio)}.mk-l{color:var(--ink);font-size:16px}
.keybox{display:flex;flex-wrap:wrap;gap:18px;font-size:13px;color:var(--ink2);margin:6px 0}.keybox .mk{margin-right:5px}
details.tv{margin:6px 0 4px}details.tv summary{cursor:pointer;color:var(--ink2);font-size:13px}
.pos{color:var(--good);font-weight:600}.neg{color:var(--ink2)}.pill{display:inline-block;padding:1px 8px;border-radius:99px;border:1px solid var(--ring);font-size:12px;color:var(--ink2)}
code{font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--surface);border:1px solid var(--ring);border-radius:4px;padding:0 4px}
.note-box{border-left:3px solid var(--muted);padding:6px 14px;margin:14px 0;background:var(--surface);color:var(--ink2);border-radius:0 8px 8px 0}
#tt{position:fixed;pointer-events:none;background:var(--ink);color:var(--bg);font-size:12.5px;padding:7px 10px;border-radius:8px;max-width:340px;z-index:9;display:none;line-height:1.4}
@media print{#tt{display:none}body{background:#fff}}
"""

JS = """
const tt=document.getElementById('tt');
document.addEventListener('mousemove',ev=>{const el=ev.target.closest('[data-tip]');if(!el){tt.style.display='none';return}
tt.innerHTML=el.dataset.tip.split(' | ').map((s,i)=>i?s:'<b>'+s+'</b>').join('<br>');tt.style.display='block';
const x=Math.min(ev.clientX+14,innerWidth-tt.offsetWidth-10),y=ev.clientY+16+tt.offsetHeight>innerHeight?ev.clientY-tt.offsetHeight-12:ev.clientY+16;tt.style.left=x+'px';tt.style.top=y+'px'});
"""


def build():
    cv = {k: (c, s, a, g) for k, _, _, c, s, a, g in IMG}
    img_rows = [[e(l), f'<span class="pill">{FAM_NAME[f]}</span>', f"{c:.3f}", f"{s:.3f}", f"{a:.3f}", (f"{g:+.3f}" if k != "clinical" else "-")]
                for k, l, f, c, s, a, g in IMG]
    parts = []
    parts.append(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Survival prediction from clinical data and lymph-node images</title><style>{CSS}</style></head><body><main>
<h1>Survival prediction from clinical data and lymph-node images</h1>
<p class="sub">Colorectal cancer, MMCI and MUG cohorts. What was built, which attributes each model used, and how well it worked.</p>
<p class="aud">Written for the project owner and collaborators. Aggregate results only: no patient-level data, dates or identifiers are in this file. Status: 27 Oct 2026.</p>

<div class="tiles">
<div class="tile"><span>Best model from the Excel sheet</span><b>0.75</b><span>cross-validated C-index (MMCI, 220 patients)</span><small>held-out test 0.74, 95% CI 0.62-0.87</small></div>
<div class="tile"><span>Images, node ratio model</span><b>0.67</b><span>C-index vs 0.62 for age + sex alone</span><small>a gain in every repeat, node and tile level; staging reference 0.74</small></div>
<div class="tile"><span>Era effect in MUG</span><b>+0.04</b><span>C-index when training only on 2000 onward</span><small>0.64 (all years) to 0.68 (from 2000)</small></div>
<div class="tile"><span>Node segmentation on MUG</span><b>91%</b><span>of nodes found after retraining, from 85%</span><small>held-out MUG slides, precision 94%</small></div>
</div>

<h2>1. Cohorts and data</h2>
<p>Two retrospective colorectal cancer cohorts. The tabular models use the clinical export; the image models use whole-slide H&amp;E images of the lymph nodes of the MMCI patients, segmented with a fine-tuned SAM3 model and embedded with Virchow2.</p>
{table(["", "MMCI", "MUG"], [
 ["Source", "Excel export, sheets 2015 and 2016", "CSV, one row per patient"],
 ["Patients used", "220 (214 have lymph-node slides)", "4,240 (after excluding 252 flagged for review)"],
 ["Deaths", "87 overall, 77 cancer-specific", "2,881 overall; no cause of death"],
 ["Median follow-up", "8.4 years", "19.4 years"],
 ["Diagnosis years", "case numbers from 2015 and 2016", "1984-2013 (follow-up to 2018)"],
 ["Stage mix", "all stages; M1 in 36 of the 202 patients with known M", "almost entirely stage II (2,084) and III (2,372); IV: 23"],
 ["Slides", "2,831 lymph-node slides, 10,729 segmented nodes", "images exist, but cannot yet be linked to patients"],
 ["Missing", "KRAS 74%, NRAS 76%, BRAF 71%, MMR 65%", "M is unknown for 87% (filled from the stage group)"],
])}
<h2>2. Attributes and where they come from</h2>
{table(["Attribute", "Source column(s)", "Derivation", "Meaning"], [[f"<b>{e(a)}</b>", e(s), e(d), e(m)] for a, s, d, m in GLOSSARY])}
""")
    parts.append(f"""
<h2>3. Which attributes each model used</h2>
<p>Every model, and exactly which attributes entered it. The slide-name suffix (LNN / LNP / M) is a pathologist's staging result and was never used by any image model.</p>
<div class="keybox"><span><span class="mk mk-c">&#9679;</span>clinical / pathology input</span><span><span class="mk mk-i">&#9670;</span>derived from the images</span>
<span><span class="mk mk-l">&#9675;</span>training label only, never an input at prediction time</span></div>
{matrix_html()}
{legend([("tab", "clinical or staging data"), ("base", "age + sex only"), ("img", "image-derived"), ("ratio", "image-derived node ratio")])}

<h2>4. MMCI: models from the Excel sheet</h2>
<p>Ridge Cox models with median imputation and dummy coding; missing categories form their own "unknown" level and levels with fewer than 10 patients are merged. Five-fold cross-validation on 176 training patients, then a one-time evaluation on a held-out test set of 44 patients (18 deaths).</p>
<h3>Cross-validated C-index by variant</h3>
{chart_tab_variants()}
{view_table(["Model", "CV C-index", "sd", "Test C-index", "Test 95% CI"], [[e(l), f"{c:.3f}", f"{s:.3f}", (f"{t:.3f}" if t else "-"), (f"{lo:.2f}-{hi:.2f}" if t else "-")] for l, f, c, s, t, lo, hi in TAB_VARIANTS])}
<p>The full model clearly beats the stage group alone (+0.08, better in all 5 folds, and the same gain on the test set). Elastic-net penalisation and keeping dMMR as its own level do not help.</p>
<h3>Which attributes matter</h3>
<p>Adjusted hazard ratios (above 1 means higher risk). Note the axis is logarithmic.</p>
{chart_forest()}
<h3>Predictive value: cross-validated loss when a feature is removed</h3>
{chart_drop_one()}
{legend([("tab", "significant after Holm correction"), ("base", "not significant")])}
<p>An asterisk marks features that are also significant after Holm correction (p &lt; 0.05). <b>M, KRAS and age</b> carry the signal. pN looks strong on its own but adds nothing once M and pT are in the model. Sex, location, grade, NRAS and BRAF add nothing detectable. MMR and KRAS partly reflect <i>who was tested</i>, since testing was mostly done in metastatic patients.</p>
""")
    parts.append(f"""
<h2>5. MUG: a large cohort with a long history</h2>
<p>The shared-feature set is what both institutes record: age, sex, pT, pN, M and grade. Because MUG spans 30 years, the choice of period matters more than the choice of stage range.</p>
<h3>Five-year survival by decade of diagnosis</h3>
{chart_era()}
<h3>Cohort choice</h3>
{chart_mug()}
{view_table(["Cohort", "Training patients", "Deaths", "CV C-index", "AUC at 5 years"], [[e(l), f"{n:,}", f"{ev:,}", f"{c:.3f}", f"{a:.3f}"] for l, n, ev, c, s, a in MUG_VARIANTS])}
<p>Training only on patients diagnosed from 2000 gives a clearly better model (0.68 vs 0.64) even with half the patients. Excluding stage IV changes nothing, because only about 18 stage IV patients are in the training data. Among the features, <b>age, pN, pT and sex</b> were significant; grade added nothing, as in MMCI. Women had a lower risk (hazard ratio 0.65 from 2000).</p>

<h2>6. Image-based survival (MMCI)</h2>
<p>Goal: predict survival without the staging from the Excel sheet, using the lymph-node images plus age and sex. 214 patients, 87 deaths, five-fold cross-validation repeated five times with the same folds for every model. Hyper-parameters were fixed in advance, not tuned on the folds.</p>
{chart_image()}
{legend([("base", "age + sex only"), ("img", "image-derived"), ("ratio", "image-derived node ratio"), ("tab", "clinical or staging data")])}
{view_table(["Model", "Family", "CV C-index", "sd", "AUC at 5 years", "Gain over age + sex"], img_rows)}
<div class="note-box"><b>What worked:</b> the lymph-node ratio approach. A node detector is trained with only "does the patient have a positive node" as a label; a patient is positive when any node is. Its expected number of positive nodes, their share of all nodes, the highest node score and the node count then enter a Cox model with age and sex. It beats age + sex by +0.033 to +0.051 C-index and by +0.06 to +0.08 AUC at five years, in every repeat. Averaging all node embeddings (+0.005) or an attention model (-0.007) did not.</div>
<p>Where it falls short: the staging reference reaches 0.74, and images with staging (0.72) are slightly below staging alone. The images capture nodal burden but not metastasis (M1) or the depth of the primary tumour, which carry most of the signal in this cohort.</p>
<h3>Do the embeddings see the staging?</h3>
{chart_staging()}
{legend([("img", "mean pooling of node embeddings"), ("ratio", "max pooling")])}
<p>Cross-validated AUC for predicting staging labels from the pooled embeddings (0.5 is chance). Node metastasis is detected very well (0.92), pT well, and M1 only moderately. This explains why the ratio approach works and why the images cannot reach the staging model on their own.</p>
<p><b>Tile-level version.</b> A detector that scores every tile (4.0 million core tiles of the 214 patients, reduced to 128 dimensions) finds patients with positive nodes better than the node-level one (AUC 0.97 vs 0.91) and its ratio matches the pathologist's more closely (Spearman 0.69 vs 0.59 among positive patients). The survival gain, however, stays the same: +0.037 to +0.044 C-index, against +0.051 for the best node-level model, all within the noise (the spread of the repeats is 0.006-0.014). Better detection of the nodes does not add survival information beyond the node-level ratio, which suggests the ratio itself is worth about +0.04 to +0.05 here.</p>

<h2>7. Lymph-node segmentation (SAM3)</h2>
<p>The fine-tuned SAM3 mask decoder was trained on MMCI slides and then tested on MUG, a different scanner and stain. It generalised in quality but missed small nodes. Retraining on MMCI plus MUG (476 MUG training slides, 135 held-out MUG test slides) fixed most of the gap.</p>
{chart_seg()}
{legend([("img", "MMCI-only decoder"), ("ratio", "MMCI + MUG decoder")])}
{table(["Held-out set", "Nodes found (IoU 0.5)", "Nodes found (IoU 0.7)", "Precision", "Slides with every node found"], [
 ["MUG test, MMCI-only decoder", "84.9%", "81.6%", "90.6%", "64%"],
 ["MUG test, MMCI + MUG decoder", "91.1%", "89.2%", "94.1%", "76%"],
 ["MMCI silver, MMCI-only decoder", "97.4%", "94.5%", "91.4%", "94%"],
 ["MMCI silver, MMCI + MUG decoder", "97.9%", "95.8%", "93.8%", "96%"],
])}
<p>Labels are produced and checked by an AI pipeline, so these numbers measure agreement with it, not with a pathologist. Nodes smaller than about 6,000 pixels (at 1/32 scale) are still found less than half the time.</p>

<h2>8. Limits and next steps</h2>
<ul>
<li><b>Small sample.</b> 87 deaths in MMCI means the C-index has an uncertainty of about 0.03. Gains below 0.05 are hard to prove; the repeated folds help, but they are not independent.</li>
<li><b>Image ratio vs the pathologist's ratio.</b> The node-level detector finds patients with positive nodes well (AUC 0.91) and the tile-level one better (0.97), but the correlation of the ratio with the pathologist's counts is moderate (Spearman 0.59 and 0.69 among positive patients). Serial sections count the same node more than once.</li>
<li><b>Training labels.</b> The ratio models use the pathologist's node counts to train the detector. They are never inputs, but a method that avoids them entirely would be weaker.</li>
<li><b>MUG images are not linked to MUG patients</b> (the CSV has only a pseudonymous patient label). A mapping would give about 20 times more patients for the image models.</li>
<li><b>Next:</b> the primary-tumour slides, which carry pT and invasion information the lymph nodes do not (the node ratio seems to be worth about +0.05, and the gap to staging is +0.12), and external data such as TCGA colorectal.</li>
</ul>

<h2>Appendix: reproducibility</h2>
{table(["Item", "Where"], [
 ["Code", "branch <code>feature/survival-prediction</code> of the ri-scale repository (preprocessing, survival_model, image_survival, scripts)"],
 ["MMCI tabular runs (MLflow experiment 72)", "preprocessing 3edf2b32, split d9092c19, Cox 38141ec7, test 3be4fedc, stage only 35e6dd5f, coxnet bb06e00f, CSS 7c219b46"],
 ["MUG runs (MLflow experiment 72)", "preprocessing 63e84dd9, split 52bc2809, cohorts bb25bb40, b12f23fb, c2fb29c8, 5d7b64be"],
 ["Image-study results", "musica, <code>$DATA/survival_features/results/</code> (core, both, ratio, tiles)"],
 ["Segmentation", "musica, <code>~/sam-ft/runs/sam3-mug-v1/</code> and <code>runs/mug-eval/</code>"],
])}
</main><div id="tt"></div><script>{JS}</script></body></html>""")
    OUT.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    build()
