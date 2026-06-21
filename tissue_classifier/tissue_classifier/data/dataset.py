from collections.abc import Iterable

import torch
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

    def __getitem__(self, _: int) -> tuple[torch.Tensor, int]:
        with OpenSlide(self.slide_path) as slide:
            thumb = slide.get_thumbnail(self.thumbnail_size).convert("RGB")
        if self.transform is not None:
            thumb = self.transform(thumb)
        return thumb, self.label


class ThumbnailDataset(MetaTiledSlides[tuple[torch.Tensor, int]]):
    """Thumbnail-level slide dataset.

    Subclasses MetaTiledSlides for MLflow artifact downloading and multi-source
    concatenation. Each slide contributes exactly one item (its thumbnail).

    Expected artifact layout per split / fold folder:
        slides.parquet   — columns: slide_path, tissue_type, case_id
        tiles.parquet    — empty (required by MetaTiledSlides; produced by save_splits)
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

    def generate_datasets(self) -> Iterable[_SlideThumbnail]:
        for slide in self.slides:
            yield _SlideThumbnail(
                slide_path=slide["slide_path"],
                label=LABEL_MAP[slide["tissue_type"]],
                thumbnail_size=self.thumbnail_size,
                transform=self.transform,
            )