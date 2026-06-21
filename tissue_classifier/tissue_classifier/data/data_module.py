from pathlib import Path

from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader

from tissue_classifier.data.dataset import (
    ThumbnailDataset,
    build_eval_transform,
    build_train_transform,
)

class DataModule(LightningDataModule):
    def __init__(
        self,
        splits_dir: str | Path,
        val_fold: int,
        batch_size: int,
        n_folds: int = 5,
        thumbnail_size: tuple[int, int] = (512, 512),
        num_workers: int = 0,
        test_on: str = "val",
    ) -> None:
        super().__init__()
        self.splits_dir = Path(splits_dir)
        self.val_fold = val_fold
        self.batch_size = batch_size
        self.n_folds = n_folds
        self.thumbnail_size = thumbnail_size
        self.num_workers = num_workers
        self.test_on = test_on

    def setup(self, stage: str) -> None:
        train_tf = build_train_transform(self.thumbnail_size)
        eval_tf = build_eval_transform(self.thumbnail_size)

        if stage == "fit":
            self._train_ds = ConcatDataset([
                ThumbnailDataset(
                    self.splits_dir / "train" / f"fold_{i}" / "slides.csv",
                    self.thumbnail_size,
                    train_tf,
                )
                for i in range(self.n_folds) if i != self.val_fold
            ])
            self._val_ds = ThumbnailDataset(
                self.splits_dir / "train" / f"fold_{self.val_fold}" / "slides.csv",
                self.thumbnail_size,
                eval_tf,
            )

        if stage in ("test", "predict"):
            self._test_ds = ThumbnailDataset(
                self.splits_dir / self.test_on / "slides.csv",
                self.thumbnail_size,
                eval_tf,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
