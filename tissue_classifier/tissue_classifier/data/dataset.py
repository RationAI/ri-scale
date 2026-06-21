from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import pyarrow as pa
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from mlflow.artifacts import download_artifacts
from rationai.mlkit.data.datasets.meta_tiled_slides import MetaTiledSlides
from openslide import OpenSlide
from torch.utils.data import Dataset
from torchvision.transforms import v2

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

LABEL_MAP = {"LN": 1, "colorectum": 0}


def build_train_transform(thumbnail_size: tuple[int, int]) -> v2.Compose:
    return v2.Compose([
        v2.Resize(thumbnail_size),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def build_eval_transform(thumbnail_size: tuple[int, int]) -> v2.Compose:
    return v2.Compose([
        v2.Resize(thumbnail_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


class _SlideThumbnail(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        slide_path: str,
        label: int,
        thumbnail_size: tuple[int, int],
        transform: v2.Compose | None,
    ) -> None:
        self.slide_path = slide_path
        self.label = label
        self.thumbnail_size = thumbnail_size
        self.transform = transform

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _: int) -> tuple[torch.Tensor, int, str]:
        with OpenSlide(self.slide_path) as slide:
            thumb = slide.get_thumbnail(self.thumbnail_size).convert("RGB")
        if self.transform is not None:
            thumb = self.transform(thumb)
        return thumb, self.label, self.slide_path


def _find_slides_parquet(p: Path) -> str | None:
    """Return path to slides.parquet whether p is the file itself or its parent folder."""
    if p.is_dir() and (p / "slides.parquet").exists():
        return str(p / "slides.parquet")
    if p.is_file() and p.name == "slides.parquet":
        return str(p)
    return None


class ThumbnailDataset(MetaTiledSlides[tuple[torch.Tensor, int]]):
    """Thumbnail-level slide dataset.

    Subclasses MetaTiledSlides for MLflow artifact downloading and
    multi-source concatenation. Each slide contributes exactly one item
    (its thumbnail). Does not require tiles.parquet.

    URIs may point to a folder containing slides.parquet or to the
    slides.parquet file directly.
    """

    def __init__(
        self,
        *,
        thumbnail_size: tuple[int, int] = (512, 512),
        transform: v2.Compose | None = None,
        **kwargs,
    ) -> None:
        self.thumbnail_size = tuple(thumbnail_size)
        self.transform = transform
        super().__init__(**kwargs)

    @staticmethod
    def load_slides_and_tiles(
        paths: Iterable[str | Path], uris: Iterable[str]
    ) -> tuple[HFDataset, HFDataset]:
        with ThreadPoolExecutor() as executor:
            artifact_paths = list(
                executor.map(lambda uri: download_artifacts(artifact_uri=uri), uris)
            )

        slide_files = []
        for raw in (*paths, *artifact_paths):
            hit = _find_slides_parquet(Path(raw))
            if hit:
                slide_files.append(hit)

        empty_tiles = HFDataset(pa.table({"slide_id": pa.array([], type=pa.string())}))

        if not slide_files:
            return HFDataset.from_dict({}), empty_tiles

        slides_ds = load_dataset("parquet", data_files=slide_files, split="train")
        return cast("HFDataset", slides_ds), empty_tiles

    def generate_datasets(self) -> Iterable[_SlideThumbnail]:
        for slide in self.slides:
            yield _SlideThumbnail(
                slide_path=slide["slide_path"],
                label=LABEL_MAP[slide["tissue_type"]],
                thumbnail_size=self.thumbnail_size,
                transform=self.transform,
            )
