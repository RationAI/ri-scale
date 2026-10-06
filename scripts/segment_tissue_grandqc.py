#!/usr/bin/env python3
"""
Segment tissue in whole-slide images with the GrandQC tissue detection model.

GrandQC (Weng et al., Nat. Commun. 2024, https://github.com/cpath-ukk/grandqc)
segments tissue vs. background with a UNet++ (EfficientNet-B0 encoder) on a slide
thumbnail at 10 µm/px, cut into 512 px tiles. This script reproduces its
inference exactly (JPEG-compressed thumbnail, tiles aligned to the image border
on the right and bottom edge) as a streaming Ray Data pipeline:

    read (CPU tasks)   open slide → thumbnail at 10 µm/px → JPEG, as GrandQC was trained on
    segment (actors)   one model per actor; the tiles of several slides share forward passes
    save (CPU tasks)   mask TIFF + overlay; returns the slide's summary row

All three stages run concurrently: slides are read on every CPU of the cluster
while the model actors (one per GPU by default) segment the ones already read.
Rows carry the thumbnail as JPEG and the mask as PNG, so moving them between
nodes is cheap.

Outputs per slide (in --out_dir):
    <slide>_mask.tiff      tissue mask (255 = tissue) at ~10 µm/px, pyramidal BigTIFF with
                           resolution tags, as written by ratiopath (xOpat, ratiopath tiling)
    <slide>_overlay.jpg    thumbnail with tissue in blue, background in gray
    summary.csv            one row per slide (failed slides have an "error")

Without --ray_address (and RAY_ADDRESS), Ray starts on the local machine. On a
cluster, slides and --out_dir must be on storage every worker sees; the weights
are loaded on the driver and shipped to the actors through the object store.

The GrandQC model is licensed CC BY-NC 4.0 (non-commercial use only).

Requirements:
    pip install "ray[data]" torch segmentation-models-pytorch openslide-python openslide-bin \
        opencv-python-headless numpy pandas pillow pyvips ratiopath

Usage:
    python segment_tissue_grandqc.py /slides/*.mrxs --out_dir masks/
    python segment_tissue_grandqc.py /mnt/data/slides/*.mrxs --out_dir /mnt/projects/masks --ray_address auto
    python segment_tissue_grandqc.py slides/*.svs --out_dir masks/ --weights Tissue_Detection_MPP10.pth
    python segment_tissue_grandqc.py slides/*.svs --out_dir masks/ --gpus_per_actor 0.5 --read_cpus 0.5
"""
import argparse
import sys
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import openslide
import pandas as pd
import pyvips
import ray
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from ratiopath.masks import write_big_tiff

_WEIGHTS_URL = "https://zenodo.org/records/14507273/files/Tissue_Detection_MPP10.pth"
_ENCODER = "timm-efficientnet-b0"
_MODEL_MPP = 10.0  # µm/px the model was trained at
_TILE = 512
_JPEG_QUALITY = 80
_TISSUE = 0  # class index; 1 is background
_TISSUE_COLOR, _BACKGROUND_COLOR = (50, 50, 250), (128, 128, 128)
_SUMMARY_COLUMNS = ["slide", "mpp", "mask_mpp", "mask_width", "mask_height",
                    "tissue_fraction", "tissue_area_mm2", "error"]


def _error(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"


# ── read ───────────────────────────────────────────────────────────────────────

def load_thumbnail(row: dict[str, Any], mpp: float | None) -> dict[str, Any]:
    """Thumbnail at the model's resolution, JPEG-compressed like GrandQC's training data."""
    path = row["path"]
    try:
        with openslide.open_slide(path) as slide:
            mpp0 = mpp or float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0) or 0)
            if mpp0 <= 0:
                raise ValueError("no resolution in the file, pass --mpp")
            width, height = slide.dimensions
            factor = _MODEL_MPP / mpp0
            thumb = slide.get_thumbnail((int(width // factor), int(height // factor)))
        # GrandQC encodes the RGB array with OpenCV, i.e. as if it were BGR. Doing the same
        # reproduces its compression artifacts exactly; decoding restores the RGB order.
        _, jpeg = cv2.imencode(".jpg", np.asarray(thumb), [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        return {
            "path": path, "mpp": mpp0, "error": "", "jpeg": jpeg.tobytes(),
            "mask_mpp_x": width * mpp0 / thumb.width, "mask_mpp_y": height * mpp0 / thumb.height,
        }
    except Exception as e:  # noqa: BLE001 -- one unreadable slide must not stop the others
        return {"path": path, "mpp": np.nan, "error": _error(e), "jpeg": b"",
                "mask_mpp_x": np.nan, "mask_mpp_y": np.nan}


def _decode(data: bytes, flags: int) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), flags)


# ── segment ────────────────────────────────────────────────────────────────────

def tile_origins(length: int) -> list[int]:
    """Tile starts along one axis: a regular grid whose last tile is aligned to the far edge."""
    starts = list(range(0, max(length - _TILE, 0) + 1, _TILE))
    if starts[-1] + _TILE < length:
        starts.append(length - _TILE)
    return starts


class TissueSegmenter:
    """Ray Data actor: GrandQC on a batch of slides, tiles of all slides batched together."""

    def __init__(self, weights: ray.ObjectRef, batch_size: int) -> None:
        # GPU if Ray assigned one (or MPS on a Mac); torch.backends.* would not pickle with the class
        self.device = torch.accelerator.current_accelerator() or torch.device("cpu")
        self.model = smp.UnetPlusPlus(encoder_name=_ENCODER, encoder_weights=None, classes=2, activation=None)
        self.model.load_state_dict(ray.get(weights))
        self.model.to(self.device).eval()
        params = smp.encoders.get_preprocessing_params(_ENCODER, "imagenet")
        self.mean = 255 * torch.tensor(params["mean"], device=self.device).view(1, 3, 1, 1)
        self.std = 255 * torch.tensor(params["std"], device=self.device).view(1, 3, 1, 1)
        self.batch_size = batch_size

    @torch.inference_mode()
    def predict(self, tiles: np.ndarray) -> np.ndarray:
        """Class map (N, H, W) for RGB uint8 tiles (N, H, W, 3)."""
        classes = []
        for start in range(0, len(tiles), self.batch_size):
            x = torch.from_numpy(tiles[start: start + self.batch_size]).to(self.device)
            x = (x.permute(0, 3, 1, 2).float() - self.mean) / self.std
            classes.append(self.model(x).argmax(dim=1).to(torch.uint8).cpu().numpy())
        return np.concatenate(classes)

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        images: dict[int, np.ndarray] = {}
        tiles, index = [], []
        for i, (jpeg, error) in enumerate(zip(batch["jpeg"], batch["error"])):
            if error:
                continue
            image = _decode(jpeg, cv2.IMREAD_COLOR)
            images[i] = image
            # Thumbnails smaller than a tile are padded with white (glass)
            pad = ((0, max(_TILE - image.shape[0], 0)), (0, max(_TILE - image.shape[1], 0)), (0, 0))
            image = np.pad(image, pad, constant_values=255)
            for y in tile_origins(image.shape[0]):
                for x in tile_origins(image.shape[1]):
                    tiles.append(image[y: y + _TILE, x: x + _TILE])
                    index.append((i, y, x))

        canvases = {i: np.empty((max(im.shape[0], _TILE), max(im.shape[1], _TILE)), np.uint8)
                    for i, im in images.items()}
        if tiles:
            # Written last-to-first, so where an edge tile overlaps its neighbour the
            # neighbour's prediction wins, as in GrandQC's stitching
            for (i, y, x), classes in zip(reversed(index), self.predict(np.stack(tiles))[::-1]):
                canvases[i][y: y + _TILE, x: x + _TILE] = classes

        masks = np.empty(len(batch["jpeg"]), dtype=object)
        for i in range(len(masks)):
            if i in canvases:
                height, width = images[i].shape[:2]
                mask = np.where(canvases[i][:height, :width] == _TISSUE, 255, 0).astype(np.uint8)
                masks[i] = cv2.imencode(".png", mask)[1].tobytes()
            else:
                masks[i] = b""
        batch["mask"] = masks
        return batch


# ── save ───────────────────────────────────────────────────────────────────────

def save_overlay(image: np.ndarray, tissue: np.ndarray, path: Path) -> None:
    colors = np.where(tissue[..., None], _TISSUE_COLOR, _BACKGROUND_COLOR).astype(np.uint8)
    Image.fromarray(cv2.addWeighted(image, 0.7, colors, 0.3, 0)).save(path, quality=_JPEG_QUALITY)


def save_outputs(row: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Writes the slide's outputs; returns its summary row."""
    path = Path(row["path"])
    summary = dict.fromkeys(_SUMMARY_COLUMNS, np.nan) | {"slide": path.name, "mpp": row["mpp"], "error": row["error"]}
    if row["error"]:
        return summary
    try:
        mask = _decode(row["mask"], cv2.IMREAD_GRAYSCALE)
        tissue = mask > 0
        write_big_tiff(pyvips.Image.new_from_array(mask), str(out_dir / f"{path.stem}_mask.tiff"),
                       mpp_x=row["mask_mpp_x"], mpp_y=row["mask_mpp_y"])
        save_overlay(_decode(row["jpeg"], cv2.IMREAD_COLOR), tissue, out_dir / f"{path.stem}_overlay.jpg")
        summary |= {
            "mask_mpp": round(row["mask_mpp_x"], 3),
            "mask_width": mask.shape[1],
            "mask_height": mask.shape[0],
            "tissue_fraction": round(float(tissue.mean()), 4),
            "tissue_area_mm2": round(tissue.sum() * row["mask_mpp_x"] * row["mask_mpp_y"] / 1e6, 2),
        }
    except Exception as e:  # noqa: BLE001
        summary["error"] = _error(e)
    return summary


# ── ray ────────────────────────────────────────────────────────────────────────

def load_weights(path: Path | None) -> dict[str, torch.Tensor]:
    if path is None:  # downloaded once into the torch hub cache
        return torch.hub.load_state_dict_from_url(
            _WEIGHTS_URL, map_location="cpu", file_name=Path(_WEIGHTS_URL).name, weights_only=True
        )
    return torch.load(path, map_location="cpu", weights_only=True)


def run(args: argparse.Namespace) -> list[dict]:
    """Streams the slides through the pipeline; returns the summary rows in completion order."""
    weights = ray.put(load_weights(args.weights))
    ds = (
        # one block per slide, so that every slide can be read by its own task
        ray.data.from_items([{"path": str(p)} for p in args.slides], override_num_blocks=len(args.slides))
        .map(partial(load_thumbnail, mpp=args.mpp), num_cpus=args.read_cpus)
        .map_batches(
            TissueSegmenter,
            fn_constructor_kwargs={"weights": weights, "batch_size": args.batch_size},
            batch_size=args.slides_per_batch,
            batch_format="numpy",
            compute=ray.data.ActorPoolStrategy(min_size=1, max_size=args.num_actors),
            num_gpus=args.gpus_per_actor,
        )
        .map(partial(save_outputs, out_dir=args.out_dir))
    )
    summary: list[dict] = []
    for row in ds.iter_rows():
        summary.append(row)
        status = f"FAILED ({row['error']})" if row["error"] else "done"
        print(f"[{len(summary)}/{len(args.slides)}] {row['slide']}: {status}", file=sys.stderr)
    return summary


# ── entrypoint ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("slides", type=Path, nargs="+", help="Slide files (anything OpenSlide reads)")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=None,
                        help="Tissue_Detection_MPP10.pth (default: downloaded from Zenodo)")
    parser.add_argument("--mpp", type=float, default=None, help="Level-0 µm/px if the file has none")
    parser.add_argument("--batch_size", type=int, default=32, help="Tiles per forward pass (default: 32)")
    parser.add_argument("--slides_per_batch", type=int, default=8,
                        help="Slides an actor segments per call, their tiles batched together (default: 8)")
    parser.add_argument("--num_actors", type=int, default=None,
                        help="Maximum number of model actors (default: one per GPU, or 1 without GPUs)")
    parser.add_argument("--gpus_per_actor", type=float, default=None,
                        help="GPUs per model actor; below 1 several actors share a GPU, overlapping their "
                             "JPEG decoding and stitching with inference (default: 1, or 0 without GPUs)")
    parser.add_argument("--read_cpus", type=float, default=1.0,
                        help="CPUs reserved per slide read; below 1 oversubscribes, for slow network "
                             "storage (default: 1)")
    parser.add_argument("--ray_address", type=str, default=None,
                        help='Ray cluster to join, e.g. "auto" (default: RAY_ADDRESS or a local instance)')
    args = parser.parse_args()

    # Ray workers do not share the driver's working directory; absolute() keeps symlinked slides' names
    args.slides = [p.absolute() for p in args.slides]
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stems = pd.Series([p.stem for p in args.slides])
    if stems.duplicated().any():
        parser.error(f"slides would overwrite each other's outputs: {sorted(set(stems[stems.duplicated()]))}")

    ray.init(address=args.ray_address)
    try:
        gpus = int(ray.cluster_resources().get("GPU", 0))
        if args.gpus_per_actor is None:
            args.gpus_per_actor = 1.0 if gpus else 0.0
        if args.num_actors is None:
            args.num_actors = max(1, int(gpus / args.gpus_per_actor) if args.gpus_per_actor else 1)
        print(f"{len(args.slides)} slides, up to {args.num_actors} model actors "
              f"with {args.gpus_per_actor:g} GPU each", file=sys.stderr)
        summary = (pd.DataFrame(run(args), columns=_SUMMARY_COLUMNS)
                   .astype({"mask_width": "Int64", "mask_height": "Int64"})  # stay integers next to failed slides
                   .sort_values("slide"))
    finally:
        ray.shutdown()
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
