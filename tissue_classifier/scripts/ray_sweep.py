"""
Multi-fold × multi-seed cross-validation sweep using Ray.

Each (fold, seed) pair runs as an independent Ray task (train + test),
producing its own MLflow run. When all tasks finish a summary MLflow run
is created with aggregated metrics and plots.

Usage:
    python scripts/ray_sweep.py \
        --folds 0 1 2 3 4 \
        --seeds 42 1337 2024 \
        --sweep-name "MMCI sweep v1" \
        --ray-address auto
"""

import argparse
import os
import subprocess
import tempfile
import time
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import ray
from mlflow.tracking import MlflowClient
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


# ── constants ────────────────────────────────────────────────────────────────

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-s3.rationai-mlflow")
EXPERIMENT_NAME = "RI Scale"
WORKDIR = str(Path(__file__).parent.parent)   # <repo>/tissue_classifier
CLASSES = ["colorectum", "LN"]
TEST_METRICS = ["test/auroc", "test/acc", "test/f1"]


# ── ray task ─────────────────────────────────────────────────────────────────

@ray.remote(num_gpus=1)
def train_fold_seed(fold: int, seed: int, sweep_start_ms: int) -> str:
    """Train + test one (fold, seed) combo. Returns the MLflow run_id."""
    env = {**os.environ, "MLFLOW_TRACKING_URI": MLFLOW_URI}
    run_name = f"fold_{fold}_seed_{seed}"

    proc = subprocess.run(
        [
            "uv", "run", "-m", "tissue_classifier",
            "+experiment=mmci",
            f"fold={fold}",
            f"seed={seed}",
            f"metadata.run_name={run_name}",
        ],
        env=env,
        cwd=WORKDIR,
    )

    if proc.returncode != 0:
        raise RuntimeError(f"Training failed: fold={fold} seed={seed}")

    # Locate the run that was just created
    from mlflow.tracking import MlflowClient as C
    client = C(MLFLOW_URI)
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=(
            f"attributes.run_name = '{run_name}' "
            f"AND attributes.start_time >= {sweep_start_ms}"
        ),
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"MLflow run not found: fold={fold} seed={seed}")
    return runs[0].info.run_id


# ── summary helpers ───────────────────────────────────────────────────────────

def _fetch_metrics(client: MlflowClient, run_id: str) -> dict[str, float]:
    run = client.get_run(run_id)
    return {k: v for k, v in run.data.metrics.items() if k.startswith("test/")}


def _fetch_predictions(client: MlflowClient, run_id: str, tmp: str) -> pd.DataFrame:
    path = client.download_artifacts(run_id, "predictions.csv", dst_path=tmp)
    return pd.read_csv(path)


def _plot_bars(df: pd.DataFrame, out: Path) -> None:
    """Per-fold grouped bars (one group per seed), one subplot per metric."""
    metrics = [m for m in TEST_METRICS if m in df.columns]
    folds = sorted(df["fold"].unique())
    seeds = sorted(df["seed"].unique())
    n_seeds = len(seeds)
    width = 0.8 / n_seeds
    colors = plt.cm.tab10.colors  # pyright: ignore[reportAttributeAccessIssue]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        for i, seed in enumerate(seeds):
            sub = df[df["seed"] == seed].set_index("fold")[metric].reindex(folds)
            x = np.arange(len(folds)) + (i - n_seeds / 2 + 0.5) * width
            ax.bar(x, sub.values, width=width * 0.9,
                   label=f"seed {seed}", color=colors[i % len(colors)], alpha=0.85)
        ax.set_xticks(np.arange(len(folds)))
        ax.set_xticklabels([f"fold {f}" for f in folds], rotation=30)
        ax.set_ylim(0, 1.05)
        ax.set_title(metric)
        ax.legend(fontsize=8)
    fig.suptitle("Test metrics per fold × seed", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_box(df: pd.DataFrame, out: Path) -> None:
    """Box plot showing distribution of each metric across all runs."""
    metrics = [m for m in TEST_METRICS if m in df.columns]
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [df[m].dropna().values for m in metrics]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], plt.cm.tab10.colors):  # pyright: ignore
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels([m.split("/")[1] for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Metric distribution across {len(df)} runs")
    ax.set_ylabel("Value")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_cm(preds_df: pd.DataFrame, out: Path) -> None:
    """Aggregated confusion matrix from all runs' predictions."""
    cm = confusion_matrix(preds_df["label"], preds_df["prediction"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"Aggregated confusion matrix — {len(preds_df)} slides")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ── summary run ───────────────────────────────────────────────────────────────

def create_summary(
    run_ids: list[str],
    job_specs: list[tuple[int, int]],
    sweep_name: str,
) -> None:
    client = MlflowClient(MLFLOW_URI)
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)

    records: list[dict] = []
    all_preds: list[pd.DataFrame] = []

    with tempfile.TemporaryDirectory() as tmp:
        for (fold, seed), run_id in zip(job_specs, run_ids):
            metrics = _fetch_metrics(client, run_id)
            records.append({"fold": fold, "seed": seed, "run_id": run_id, **metrics})
            try:
                preds = _fetch_predictions(client, run_id, tmp)
                preds["fold"] = fold
                preds["seed"] = seed
                all_preds.append(preds)
            except Exception as e:
                print(f"[warn] Could not fetch predictions for {run_id}: {e}")

        df = pd.DataFrame(records)
        preds_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()

        # ── aggregate metrics ────────────────────────────────────────────────
        summary_metrics: dict[str, float] = {}
        for metric in TEST_METRICS:
            if metric not in df.columns:
                continue
            vals = df[metric].dropna()
            summary_metrics[f"{metric}/mean"] = float(vals.mean())
            summary_metrics[f"{metric}/std"] = float(vals.std())
            summary_metrics[f"{metric}/min"] = float(vals.min())
            summary_metrics[f"{metric}/max"] = float(vals.max())

        # ── plots ────────────────────────────────────────────────────────────
        artifacts: list[str] = []

        bar_path = Path(tmp) / "metrics_per_fold.png"
        _plot_bars(df, bar_path)
        artifacts.append(str(bar_path))

        box_path = Path(tmp) / "metrics_distribution.png"
        _plot_box(df, box_path)
        artifacts.append(str(box_path))

        if not preds_df.empty:
            cm_path = Path(tmp) / "confusion_matrix_aggregated.png"
            _plot_cm(preds_df, cm_path)
            artifacts.append(str(cm_path))

            merged_csv = Path(tmp) / "predictions_all_runs.csv"
            preds_df.to_csv(merged_csv, index=False)
            artifacts.append(str(merged_csv))

        metrics_csv = Path(tmp) / "metrics_summary.csv"
        df.to_csv(metrics_csv, index=False)
        artifacts.append(str(metrics_csv))

        # ── create summary MLflow run ────────────────────────────────────────
        mlflow.set_tracking_uri(MLFLOW_URI)
        with mlflow.start_run(experiment_id=exp.experiment_id, run_name=sweep_name) as run:
            mlflow.log_params({
                "folds": str(sorted(df["fold"].unique().tolist())),
                "seeds": str(sorted(df["seed"].unique().tolist())),
                "n_runs": len(run_ids),
            })
            mlflow.log_metrics(summary_metrics)
            for artifact in artifacts:
                mlflow.log_artifact(artifact)

        url = f"{MLFLOW_URI}/#/experiments/{exp.experiment_id}/runs/{run.info.run_id}"
        print(f"\nSummary run: {url}")


# ── entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ray cross-validation sweep.")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)),
                        help="Fold indices to run (default: 0 1 2 3 4)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Seeds to run per fold (default: 42)")
    parser.add_argument("--sweep-name", type=str, default="Sweep Summary",
                        help="MLflow run name for the summary run")
    parser.add_argument("--ray-address", type=str, default="auto",
                        help="Ray cluster address (default: auto)")
    args = parser.parse_args()

    ray.init(address=args.ray_address)

    job_specs = list(product(args.folds, args.seeds))
    sweep_start_ms = int(time.time() * 1000)

    print(f"Submitting {len(job_specs)} jobs "
          f"({len(args.folds)} folds × {len(args.seeds)} seeds)…")

    futures = [
        train_fold_seed.remote(fold, seed, sweep_start_ms)
        for fold, seed in job_specs
    ]
    run_ids: list[str] = ray.get(futures)

    print(f"\nAll {len(run_ids)} jobs completed. Building summary…")
    create_summary(run_ids, job_specs, args.sweep_name)

    ray.shutdown()


if __name__ == "__main__":
    main()
