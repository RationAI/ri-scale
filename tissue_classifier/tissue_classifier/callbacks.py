import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from lightning import Callback, LightningModule, Trainer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


_CLASSES = ["colorectum", "LN"]


class PredictionLogger(Callback):
    """Saves per-slide test predictions to CSV and confusion matrix PNG, uploads both to MLflow."""

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._records: list[dict] = []

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        for path, prob, label in zip(outputs["paths"], outputs["probs"], outputs["labels"]):
            prob_val = prob.item()
            self._records.append({
                "slide_name": Path(path).name,
                "slide_path": path,
                "probability": round(prob_val, 6),
                "confidence": round(max(prob_val, 1.0 - prob_val), 6),
                "prediction": int(prob_val > 0.5),
                "label": int(label.item()),
            })

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        df = pd.DataFrame(self._records)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            df.to_csv(tmp / "predictions.csv", index=False)
            self._save_confusion_matrix(df, tmp / "confusion_matrix.png")
            client = trainer.logger.experiment
            run_id = trainer.logger.run_id
            client.log_artifact(run_id, str(tmp / "predictions.csv"))
            client.log_artifact(run_id, str(tmp / "confusion_matrix.png"))

    @staticmethod
    def _save_confusion_matrix(df: pd.DataFrame, path: Path) -> None:
        cm = confusion_matrix(df["label"], df["prediction"])
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=_CLASSES).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title("Test — confusion matrix")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
