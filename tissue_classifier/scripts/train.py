import hydra
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from tissue_classifier.data.data_module import DataModule
from tissue_classifier.meta_arch import MetaArch
from tissue_classifier.modeling.backbone import Backbone
from tissue_classifier.modeling.head import ClassifierHead


@hydra.main(
    config_path="../configs",
    config_name="base",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    backbone = Backbone(
        model_name=config.model.backbone_name,
        pretrained=config.model.pretrained,
    )
    head = ClassifierHead(in_features=backbone.feature_dim)
    model = MetaArch(
        backbone=backbone,
        head=head,
        lr=float(config.training.lr),
        weight_decay=float(config.training.weight_decay),
    )

    thumbnail_size = tuple(config.model.thumbnail_size)
    datamodule = DataModule(
        splits_dir=config.splits_dir,
        val_fold=int(config.fold),
        batch_size=int(config.training.batch_size),
        thumbnail_size=thumbnail_size,
        num_workers=int(config.training.num_workers),
        test_on="val",
    )

    callbacks = [
        ModelCheckpoint(monitor="val/auroc", mode="max", save_top_k=1, filename="best"),
        EarlyStopping(monitor="val/auroc", mode="max", patience=10),
    ]

    trainer = Trainer(
        max_epochs=config.training.max_epochs,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
