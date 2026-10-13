"""Usage: uv run scripts/train.py data.splits_uri=mlflow-artifacts:/<exp>/<run>/artifacts [+experiment=mug] [model=rsf ...]"""

import shlex
import sys

from kube_jobs import storage, submit_job

# Hydra overrides from the command line, so run URIs need no commit
args = sys.argv[1:]
if not any(arg.startswith("+experiment=") for arg in args):
    args = ["+experiment=mmci", *args]  # MMCI unless another dataset is given
overrides = " ".join(shlex.quote(arg) for arg in args)


submit_job(
    job_name="ri-scale-survival-model-train",
    username="pekarj",
    cpu=8,
    memory="16Gi",
    gpu=None,
    public=False,
    script=[
        "git clone --single-branch --branch feature/survival-prediction https://github.com/RationAI/ri-scale.git workdir",
        "cd workdir/survival_model",
        "uv sync --frozen",
        "export MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow",
        f"uv run -m survival_model {overrides}",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
