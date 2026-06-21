import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from openslide import OpenSlide
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from tqdm import tqdm

from preprocessing.data_source import DataSource
from preprocessing.name_parsers import ParsedFilename, parse_filename

log = logging.getLogger(__name__)


# ── slide processing ───────────────────────────────────────────────────────────

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


def build_slides_df(source: DataSource, max_workers: int = 4) -> pd.DataFrame:
    paths = list(source)
    if not paths:
        raise ValueError("Data source returned no slide paths")

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


# ── entrypoint ─────────────────────────────────────────────────────────────────

@with_cli_args(["+preprocessing=patient_dataset"])
@hydra.main(
    config_path="../configs",
    config_name="preprocessing",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    source: DataSource = hydra.utils.instantiate(config.dataset.slides)
    slides_df = build_slides_df(source, max_workers=config.max_workers)
    patients_df = build_patients_df(slides_df)

    active = slides_df[~slides_df["excluded"]]
    mlflow.log_metrics({
        "n_slides_total": len(slides_df),
        "n_slides_active": len(active),
        "n_slides_excluded": int(slides_df["excluded"].sum()),
        "n_slides_ln": int((active["tissue_type"] == "LN").sum()),
        "n_slides_colorectum": int((active["tissue_type"] == "colorectum").sum()),
        "n_cases_total": int(active["case_id"].nunique()),
    })

    with tempfile.TemporaryDirectory() as tmp_dir:
        slides_df.to_csv(f"{tmp_dir}/slides.csv", index=False)
        patients_df.to_csv(f"{tmp_dir}/patients.csv", index=False)
        logger.log_artifacts(tmp_dir)

    mlflow.log_input(
        mlflow.data.from_pandas(slides_df, name="slides"),
        context="slides",
    )
    mlflow.log_input(
        mlflow.data.from_pandas(patients_df, name="patients"),
        context="patients",
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
