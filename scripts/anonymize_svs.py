#!/usr/bin/env python3
"""
Anonymize SVS slides from a selection CSV:
  - Strips all associated images (thumbnail / label / macro) that may contain barcodes.
  - Keeps only pyramid levels at ~0.5 µm/px or coarser (removes the high-res level).
  - Renames each slide to a random UUID-based filename.
  - Writes a name-mapping CSV: original_name, new_name.

The output files are valid pyramidal TIFFs readable by OpenSlide.

Requirements:
    pip install pyvips openslide-python pandas
    # libvips must also be installed at the system level:
    #   macOS:  brew install vips
    #   Ubuntu: apt-get install libvips-dev

Usage:
    python anonymize_svs.py selection.csv /output/dir
    python anonymize_svs.py selection.csv /output/dir --target_mpp 0.5 --input_dir /slides
"""
import argparse
import sys
import uuid
from pathlib import Path

import openslide
import pandas as pd
import pyvips


# ── helpers ────────────────────────────────────────────────────────────────────

def _get_target_level(svs_path: Path, target_mpp: float) -> int:
    """Return the first pyramid level whose effective MPP >= target_mpp."""
    with openslide.OpenSlide(str(svs_path)) as slide:
        base_mpp = float(slide.properties.get("openslide.mpp-x", 0.0))
        downsamples = list(slide.level_downsamples)

    if base_mpp <= 0:
        print(f"  Warning: cannot determine MPP for {svs_path.name}, using level 0", file=sys.stderr)
        return 0

    for level, ds in enumerate(downsamples):
        if base_mpp * ds >= target_mpp:
            return level

    # All levels are finer than target_mpp — fall back to coarsest available
    return len(downsamples) - 1


def _get_level_mpp(svs_path: Path, level: int) -> tuple[float, float]:
    with openslide.OpenSlide(str(svs_path)) as slide:
        mpp_x = float(slide.properties.get("openslide.mpp-x", 0.0))
        mpp_y = float(slide.properties.get("openslide.mpp-y", 0.0))
        ds = slide.level_downsamples[level]
    return mpp_x * ds, mpp_y * ds


# ── core processing ────────────────────────────────────────────────────────────

def process_slide(input_path: Path, output_path: Path, target_mpp: float) -> None:
    """
    Load the slide at the first level with MPP >= target_mpp using pyvips.

    pyvips' openslide loader loads only the main pyramid image and does NOT
    include associated images (thumbnail / label / macro), so barcodes are
    excluded automatically.  The output is a tiled, pyramidal TIFF.
    """
    target_level = _get_target_level(input_path, target_mpp)
    mpp_x, mpp_y = _get_level_mpp(input_path, target_level)

    img = pyvips.Image.new_from_file(
        str(input_path),
        level=target_level,
        access="sequential",
    )

    # Resolution in pixels per cm (1 cm = 10 000 µm)
    xres = (10_000.0 / mpp_x) if mpp_x > 0 else 0.0
    yres = (10_000.0 / mpp_y) if mpp_y > 0 else 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.tiffsave(
        str(output_path),
        tile=True,
        pyramid=True,
        compression="jpeg",
        Q=70,
        tile_width=256,
        tile_height=256,
        bigtiff=True,
        xres=xres,
        yres=yres,
        resunit=pyvips.enums.ForeignTiffResunit.CM,
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove thumbnails/barcodes and high-res levels from SVS slides, then rename."
    )
    parser.add_argument(
        "selection_csv",
        type=Path,
        help="CSV produced by select_slides.py (must have 'slide_name' and 'original_path' columns).",
    )
    parser.add_argument("output_dir", type=Path, help="Directory to write anonymized SVS files.")
    parser.add_argument(
        "--mapping_csv",
        type=Path,
        default=None,
        help="Where to write the name-mapping CSV (default: <output_dir>/name_mapping.csv).",
    )
    parser.add_argument(
        "--target_mpp",
        type=float,
        default=0.5,
        help="Keep levels with effective MPP >= this value in µm/px (default: 0.5).",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=None,
        help="If set, look for <slide_name> inside this directory instead of using 'original_path'.",
    )
    args = parser.parse_args()

    mapping_csv = args.mapping_csv or args.output_dir / "name_mapping.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selection = pd.read_csv(args.selection_csv)
    required = {"slide_name", "original_path"}
    missing = required - set(selection.columns)
    if missing:
        sys.exit(f"Selection CSV is missing required columns: {missing}")

    mapping_rows = []
    errors = []

    for i, row in selection.iterrows():
        if args.input_dir:
            input_path = args.input_dir / row["slide_name"]
        else:
            input_path = Path(row["original_path"])

        if not input_path.exists():
            print(f"  [{i+1}/{len(selection)}] SKIP  {row['slide_name']} — file not found", file=sys.stderr)
            errors.append(row["slide_name"])
            continue

        new_name = f"{uuid.uuid4().hex}.svs"
        output_path = args.output_dir / new_name

        print(f"  [{i+1}/{len(selection)}] {input_path.name}  →  {new_name}")
        try:
            process_slide(input_path, output_path, args.target_mpp)
        except Exception as exc:
            print(f"    ERROR: {exc}", file=sys.stderr)
            errors.append(row["slide_name"])
            continue

        mapping_rows.append({
            "original_name": row["slide_name"],
            "original_path": str(input_path),
            "new_name": new_name,
            "new_path": str(output_path),
            "case_id": row.get("case_id", ""),
            "slide_id": row.get("slide_id", ""),
        })

    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df.to_csv(mapping_csv, index=False)

    print(f"\nDone. {len(mapping_rows)} slides processed, {len(errors)} skipped.")
    print(f"Name mapping → {mapping_csv}")
    if errors:
        print(f"Failed/skipped: {errors}", file=sys.stderr)


if __name__ == "__main__":
    main()
