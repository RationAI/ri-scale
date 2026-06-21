import hydra
from hydra.utils import instantiate
from lightning import seed_everything
from omegaconf import DictConfig
from rationai.mlkit import Trainer, autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger

from tissue_classifier.data.data_module import DataModule
from tissue_classifier.meta_arch import MetaArch


@hydra.main(config_path="../configs", config_name="base", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    seed_everything(config.seed, workers=True)

    data = instantiate(config.data, _recursive_=False, _target_=DataModule)
    model = instantiate(config.model, _target_=MetaArch)

    trainer = instantiate(config.trainer, _target_=Trainer, logger=logger)
    getattr(trainer, config.mode)(model, datamodule=data, ckpt_path=config.checkpoint)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
