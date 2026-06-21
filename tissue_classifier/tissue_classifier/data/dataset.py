from pathlib import Path

import pandas as pd
import torch
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


class InferenceDataset(Dataset):
    """Dataset for unlabeled slides — returns (tensor, slide_path) pairs."""

    def __init__(
        self,
        slide_paths: list[str],
        thumbnail_size: tuple[int, int] = (512, 512),
        transform: v2.Compose | None = None,
    ) -> None:
        self.slide_paths = slide_paths
        self.thumbnail_size = thumbnail_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.slide_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        path = self.slide_paths[idx]
        with OpenSlide(path) as slide:
            thumb = slide.get_thumbnail(self.thumbnail_size).convert("RGB")
        if self.transform is not None:
            thumb = self.transform(thumb)
        return thumb, path


class ThumbnailDataset(Dataset):
    def __init__(
        self,
        slides_csv: Path | str,
        thumbnail_size: tuple[int, int] = (512, 512),
        transform: v2.Compose | None = None,
    ) -> None:
        self.slides = pd.read_csv(slides_csv)
        self.thumbnail_size = thumbnail_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.slides.iloc[idx]
        with OpenSlide(row["slide_path"]) as slide:
            thumb = slide.get_thumbnail(self.thumbnail_size).convert("RGB")
        if self.transform is not None:
            thumb = self.transform(thumb)
        return thumb, LABEL_MAP[row["tissue_type"]]
