#!/usr/bin/env python3
"""
Detect stitching seams in whole-slide images.

A scanner images the slide as a grid of overlapping camera fields and stitches
them. When stitching fails (typically in fatty tissue, which has too little
texture to align on), field borders show up as straight horizontal / vertical
lines across which the image jumps: content is shifted, brightness or colour
changes, or focus changes.

Detection: for every row (and column) boundary, the intensity jump across it is
compared with the jumps between the neighbouring rows, relative to the local
texture. Natural edges are spread over several pixels and are not straight for
long; a seam is a one-pixel discontinuity along a straight line. Lines at least
``--min_seam_um`` long are reported.

Outputs per slide (in --out_dir):
    <slide>_tiles.csv      per tissue tile: longest seams, edge contrast, flag
    <slide>_seams.csv      every detected seam line in level-0 coordinates
    <slide>_overlay.png    thumbnail with seam lines (red) and flagged tiles
    <slide>.geojson        seam lines and flagged tiles, importable to QuPath / xOpat
    summary.csv            one row per slide (failed slides have an "error")

Every slide is a Ray task reserving --cpus_per_slide CPUs, which it uses to process
its tiles in parallel. Without --ray_address (and RAY_ADDRESS), Ray starts on the
local machine. On a cluster, slides and --out_dir must be on storage every worker sees.

Requirements:
    pip install ray openslide-python openslide-bin numpy scipy pandas pillow

Usage:
    python detect_stitching_artifacts.py /slides/*.mrxs --out_dir qc/
    python detect_stitching_artifacts.py /mnt/data/slides/*.mrxs --out_dir /mnt/projects/qc --ray_address auto
    python detect_stitching_artifacts.py slide.svs --out_dir qc/ --target_mpp 0.5 --min_seam_um 150
    python detect_stitching_artifacts.py screenshot.png --out_dir qc/ --mpp 0.27   # plain images need --mpp
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import openslide
import pandas as pd
import ray
from PIL import Image, ImageDraw
from scipy import ndimage

_GRAY = np.array([0.299, 0.587, 0.114], dtype=np.float32)
_MARGIN = 4  # px read around each tile so that seams on tile borders are seen


# ── seam detection ─────────────────────────────────────────────────────────────

def _segment_mean(a: np.ndarray, seg: int) -> np.ndarray:
    n = a.shape[1] // seg
    return a[:, : n * seg].reshape(a.shape[0], n, seg).mean(axis=2)


def discontinuity(gray: np.ndarray, seg: int) -> np.ndarray:
    """Relative jump across each row boundary, per segment of ``seg`` columns.

    Row i is the boundary between image rows i and i+1. The jump D across it is
    compared with B, the jumps two rows away on either side: (D - B) / (B + 1).
    About 0 inside smooth or textured tissue, large across a seam. The jump is
    taken over two adjacent boundaries because a seam can straddle two pixels
    in a downsampled pyramid level. Transparent (not scanned) pixels are NaN and
    never produce a seam.
    """
    d = _segment_mean(np.abs(np.diff(gray, axis=0)), seg)
    jump = np.maximum(d[2:-2], d[3:-1])
    base = 0.5 * (d[:-4] + d[4:])
    return np.pad((jump - base) / (base + 1.0), ((2, 2), (0, 0)), constant_values=np.nan)


def _longest_run(hit: np.ndarray, gap: int) -> tuple[int, int, int]:
    """(length, start, end) of the longest run of hits, allowing ``gap`` misses inside it."""
    best = (0, 0, 0)
    start, run, miss = 0, 0, 0
    for i, h in enumerate(hit):
        if h:
            if run == 0:
                start = i
            run, miss = run + 1 + miss, 0
            best = max(best, (run, start, i + 1))
        elif run and miss < gap:
            miss += 1
        else:
            run, miss = 0, 0
    return best


def find_lines(
    gray: np.ndarray, seg: int, threshold: float, min_segments: int, gap: int, core: tuple[int, int]
) -> list[tuple[int, int, int, float]]:
    """Horizontal seams as (row, first column, last column, mean score); rows limited to ``core``."""
    scores = discontinuity(gray, seg)
    candidates = []
    for row in range(*core):
        hit = np.nan_to_num(scores[row], nan=0.0) > threshold
        length, start, end = _longest_run(hit, gap)
        if length < min_segments:
            continue
        score = float(np.nanmean(scores[row, start:end]))
        if score >= threshold:  # the gaps allowed inside a run must not carry a weak line
            candidates.append((length, row, start, end, score))

    # The same seam lights up on adjacent rows; keep the longest one
    kept: list[tuple[int, int, int, float]] = []
    for _, row, start, end, score in sorted(candidates, reverse=True):
        if all(abs(row - r) > 3 or end <= s or start >= e for r, s, e, _ in kept):
            kept.append((row, start, end, score))
    return [(row, start * seg, end * seg, score) for row, start, end, score in kept]


def edge_contrast(gray: np.ndarray) -> float:
    """Mean gradient of the strongest 2 % edges. Informational: a drop relative to
    neighbouring tiles of the same tissue can mean defocus or haze."""
    valid = np.nan_to_num(gray, nan=255.0)
    grad = ndimage.gaussian_gradient_magnitude(valid, 0.7)
    return float(grad[grad >= np.quantile(grad, 0.98)].mean())


# ── slide processing ───────────────────────────────────────────────────────────

def tissue_mask(slide: openslide.AbstractSlide, mpp0: float, mask_mpp: float) -> tuple[np.ndarray, float]:
    """Coarse tissue mask (fat cells filled in) and its downsample w.r.t. level 0."""
    downsample = mask_mpp / mpp0
    width, height = slide.dimensions
    thumb = slide.get_thumbnail((max(1, int(width / downsample)), max(1, int(height / downsample))))
    downsample = slide.dimensions[0] / thumb.width
    saturation = np.asarray(thumb.convert("HSV"))[..., 1]
    mask = saturation > 12
    radius = max(1, round(64 / mask_mpp))  # close gaps of up to ~ one fat cell
    structure = ndimage.iterate_structure(ndimage.generate_binary_structure(2, 1), radius)
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask, structure))
    return mask, downsample


def process_tile(slide: openslide.AbstractSlide, level: int, x: int, y: int, args: argparse.Namespace,
                 tile: int, downsample: float, mpp: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Seams inside one tile (x, y: level-0 coordinates of the tile's top-left corner)."""
    size = tile + 2 * _MARGIN
    origin = (int(x - _MARGIN * downsample), int(y - _MARGIN * downsample))
    rgba = np.asarray(slide.read_region(origin, level, (size, size)), dtype=np.float32)
    gray = rgba[..., :3] @ _GRAY
    gray[rgba[..., 3] == 0] = np.nan

    seg = max(4, round(args.segment_um / mpp))
    min_segments = max(1, round(args.min_seam_um / (seg * mpp)))
    core = (_MARGIN - 1, _MARGIN + tile - 1)

    seams: list[dict[str, Any]] = []
    longest = {"horizontal": 0.0, "vertical": 0.0}
    for orientation, image in (("horizontal", gray), ("vertical", gray.T)):
        for line, start, end, score in find_lines(image, seg, args.threshold, min_segments, args.gap, core):
            # boundary between image rows `line` and `line + 1`, in level-0 coordinates
            across = origin[1 if orientation == "horizontal" else 0] + (line + 1) * downsample
            along = origin[0 if orientation == "horizontal" else 1]
            a0, a1 = along + start * downsample, along + end * downsample
            x0, y0, x1, y1 = (a0, across, a1, across) if orientation == "horizontal" else (across, a0, across, a1)
            seams.append({
                "orientation": orientation,
                "x0": round(x0), "y0": round(y0), "x1": round(x1), "y1": round(y1),
                "length_um": round((end - start) * mpp, 1),
                "score": round(score, 2),
            })
            longest[orientation] = max(longest[orientation], (end - start) * mpp)

    record = {
        "x": x, "y": y, "size": round(tile * downsample),
        "longest_horizontal_um": round(longest["horizontal"], 1),
        "longest_vertical_um": round(longest["vertical"], 1),
        "n_seams": len(seams),
        "edge_contrast": round(edge_contrast(gray), 2),
        "seam": bool(seams),
    }
    return record, seams


def process_slide(path: Path, args: argparse.Namespace) -> dict:
    slide = openslide.open_slide(str(path))
    mpp0 = args.mpp or float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0) or 0)
    if mpp0 <= 0:
        raise ValueError(f"{path.name}: no resolution in the file, pass --mpp")

    level = slide.get_best_level_for_downsample(args.target_mpp / mpp0)
    downsample = slide.level_downsamples[level]
    mpp = mpp0 * downsample
    tile = args.tile_size
    step = tile * downsample  # level-0 px per tile

    mask, mask_downsample = tissue_mask(slide, mpp0, args.mask_mpp)
    positions = []
    for y in np.arange(0, slide.dimensions[1], step):
        for x in np.arange(0, slide.dimensions[0], step):
            window = mask[int(y / mask_downsample): int((y + step) / mask_downsample) + 1,
                          int(x / mask_downsample): int((x + step) / mask_downsample) + 1]
            if window.size and window.mean() >= args.min_tissue:
                positions.append((int(x), int(y)))
    print(f"{path.name}: level {level} ({mpp:.2f} µm/px), {len(positions)} tissue tiles", file=sys.stderr)

    with ThreadPoolExecutor(args.cpus_per_slide) as executor:
        results = list(executor.map(
            lambda p: process_tile(slide, level, p[0], p[1], args, tile, downsample, mpp), positions
        ))
    tiles = pd.DataFrame([r for r, _ in results])
    seams = pd.DataFrame([s for _, tile_seams in results for s in tile_seams],
                         columns=["orientation", "x0", "y0", "x1", "y1", "length_um", "score"])

    stem = path.stem
    tiles.to_csv(args.out_dir / f"{stem}_tiles.csv", index=False)
    seams.to_csv(args.out_dir / f"{stem}_seams.csv", index=False)
    save_overlay(slide, tiles, seams, args.out_dir / f"{stem}_overlay.png")
    save_geojson(tiles, seams, args.out_dir / f"{stem}.geojson")
    slide.close()

    n_flagged = int(tiles["seam"].sum()) if len(tiles) else 0
    return {
        "slide": path.name,
        "mpp": round(mpp, 3),
        "n_tissue_tiles": len(tiles),
        "n_seam_tiles": n_flagged,
        "seam_tile_fraction": round(n_flagged / len(tiles), 4) if len(tiles) else 0.0,
        "total_seam_mm": round(seams["length_um"].sum() / 1000, 2) if len(seams) else 0.0,
    }


# ── outputs ────────────────────────────────────────────────────────────────────

def save_overlay(slide: openslide.AbstractSlide, tiles: pd.DataFrame, seams: pd.DataFrame, path: Path) -> None:
    thumb = slide.get_thumbnail((2048, 2048)).convert("RGB")
    scale = thumb.width / slide.dimensions[0]
    shade = Image.new("RGBA", thumb.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(shade)
    for t in tiles[tiles["seam"]].itertuples() if len(tiles) else []:
        draw.rectangle([t.x * scale, t.y * scale, (t.x + t.size) * scale, (t.y + t.size) * scale],
                       fill=(255, 200, 0, 70), outline=(255, 150, 0, 255))
    for s in seams.itertuples():
        draw.line([s.x0 * scale, s.y0 * scale, s.x1 * scale, s.y1 * scale], fill=(220, 0, 0, 255), width=2)
    Image.alpha_composite(thumb.convert("RGBA"), shade).convert("RGB").save(path)


def save_geojson(tiles: pd.DataFrame, seams: pd.DataFrame, path: Path) -> None:
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[s.x0, s.y0], [s.x1, s.y1]]},
            "properties": {"objectType": "annotation", "classification": {"name": "Stitching seam"},
                           "length_um": s.length_um, "score": s.score},
        }
        for s in seams.itertuples()
    ]
    for t in tiles[tiles["seam"]].itertuples() if len(tiles) else []:
        x0, y0, x1, y1 = t.x, t.y, t.x + t.size, t.y + t.size
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]},
            "properties": {"objectType": "annotation", "classification": {"name": "Seam tile"}},
        })
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))


# ── ray ────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> list[dict]:
    """One Ray task per slide; returns the summary rows in completion order."""
    task = ray.remote(process_slide).options(num_cpus=args.cpus_per_slide)
    pending = {task.remote(path, args): path for path in args.slides}
    summary: list[dict] = []
    while pending:
        [done], _ = ray.wait(list(pending), num_returns=1)
        path = pending.pop(done)
        try:
            summary.append(ray.get(done))
            status = "done"
        except ray.exceptions.RayError as e:
            error = getattr(e, "cause", None) or e  # the slide's own exception, not the Ray traceback
            summary.append({"slide": path.name, "error": f"{type(error).__name__}: {error}"})
            status = f"FAILED ({type(error).__name__})"
        print(f"[{len(summary)}/{len(args.slides)}] {path.name}: {status}", file=sys.stderr)
    return summary


# ── entrypoint ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("slides", type=Path, nargs="+", help="Slide files (anything OpenSlide reads)")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--target_mpp", type=float, default=0.5,
                        help="Analysis resolution; the closest finer pyramid level is used (default: 0.5)")
    parser.add_argument("--mpp", type=float, default=None, help="Level-0 µm/px if the file has none")
    parser.add_argument("--tile_size", type=int, default=1024, help="Tile size in px at the analysis level")
    parser.add_argument("--min_tissue", type=float, default=0.25, help="Minimum tissue fraction of a tile")
    parser.add_argument("--mask_mpp", type=float, default=8.0, help="Resolution of the tissue mask")
    parser.add_argument("--segment_um", type=float, default=16.0, help="Length of the pieces a line is scored in")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Relative jump a piece needs to count as seam (default: 1.0)")
    parser.add_argument("--min_seam_um", type=float, default=100.0, help="Minimum seam length (default: 100)")
    parser.add_argument("--gap", type=int, default=1, help="Pieces below threshold allowed inside a seam")
    parser.add_argument("--cpus_per_slide", type=int, default=4,
                        help="CPUs reserved per Ray task, also the tile threads per slide (default: 4)")
    parser.add_argument("--ray_address", type=str, default=None,
                        help='Ray cluster to join, e.g. "auto" (default: RAY_ADDRESS or a local instance)')
    args = parser.parse_args()

    # Ray workers do not share the driver's working directory
    args.slides = [p.resolve() for p in args.slides]
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ray.init(address=args.ray_address)
    try:
        summary = pd.DataFrame(run(args)).sort_values("slide")
    finally:
        ray.shutdown()
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
