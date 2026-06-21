from collections.abc import Iterable

from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader


class DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 0,
        **datasets: DictConfig,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datasets = datasets

    def setup(self, stage: str) -> None:
        match stage:
            case "fit":
                self._train_ds = instantiate(self.datasets["train"])
                self._val_ds = instantiate(self.datasets["val"])
            case "validate":
                self._val_ds = instantiate(self.datasets["val"])
            case "test" | "predict":
                self._test_ds = instantiate(self.datasets["test"])

    def train_dataloader(self) -> Iterable:
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Iterable:
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> Iterable:
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
