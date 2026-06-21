import torch
from lightning import LightningModule
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryF1Score

from tissue_classifier.modeling.backbone import Backbone
from tissue_classifier.modeling.head import ClassifierHead


class MetaArch(LightningModule):
    def __init__(
        self,
        backbone: Backbone,
        head: ClassifierHead,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.criterion = nn.BCEWithLogitsLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=["backbone", "head"])

        metrics = MetricCollection({
            "auroc": BinaryAUROC(),
            "acc": BinaryAccuracy(),
            "f1": BinaryF1Score(),
        })
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.backbone(x))

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.val_metrics.update(logits, y)

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute(), on_epoch=True)
        self.val_metrics.reset()

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        self.test_metrics.update(logits, y)

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute(), on_epoch=True)
        self.test_metrics.reset()

    def predict_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> Tensor:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return torch.sigmoid(self(x))

    def configure_optimizers(self) -> dict:
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
