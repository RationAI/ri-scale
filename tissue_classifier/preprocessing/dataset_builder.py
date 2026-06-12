import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from openslide import OpenSlide
from tqdm import tqdm

from preprocessing.name_parsers import ParsedFilename, parse_filename

log = logging.getLogger(__name__)


def _read_slide_metadata(slide_path: str) -> dict:
    with OpenSlide(slide_path) as slide:
        return {
            "mpp_x": float(slide.properties.get("openslide.mpp-x", float("nan"))),
            "mpp_y": float(slide.properties.get("openslide.mpp-y", float("nan"))),
            "n_levels": slide.level_count,
            "vendor": slide.properties.get("openslide.vendor"),
        }


def _process_slide(path: str) -> dict | None:
    try:
        parsed: ParsedFilename = parse_filename(Path(path).name)
    except ValueError:
        log.warning("Skipping unrecognised filename: %s", path)
        return None

    meta = (
        {"mpp_x": float("nan"), "mpp_y": float("nan"), "n_levels": None, "vendor": None}
        if parsed.excluded
        else _read_slide_metadata(path)
    )

    return {
        "slide_path": path,
        "slide_name": Path(path).stem,
        "case_id": parsed.case_id,
        "slide_num": parsed.slide_num,
        "label": parsed.label,
        "tissue_type": parsed.tissue_type,
        "excluded": parsed.excluded,
        **meta,
    }


def build_slides_df(slides_dir: Path, max_workers: int = 4) -> pd.DataFrame:
    paths = sorted(str(p) for p in slides_dir.glob("*.mrxs"))
    if not paths:
        raise FileNotFoundError(f"No .mrxs files found in {slides_dir}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        rows = list(
            tqdm(executor.map(_process_slide, paths), total=len(paths), desc="Reading slides")
        )

    rows = [r for r in rows if r is not None]
    return pd.DataFrame(rows)


def build_patients_df(slides_df: pd.DataFrame) -> pd.DataFrame:
    active = slides_df[~slides_df["excluded"]]
    return (
        active.groupby("case_id")
        .agg(
            n_slides=("slide_path", "count"),
            n_ln_slides=("tissue_type", lambda x: (x == "LN").sum()),
            n_colorectum_slides=("tissue_type", lambda x: (x == "colorectum").sum()),
        )
        .reset_index()
    )
