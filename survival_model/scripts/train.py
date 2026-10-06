"""Usage: uv run scripts/train.py data.splits_uri=mlflow-artifacts:/<exp>/<run>/artifacts [model=rsf ...]"""

import shlex
import sys

from kube_jobs import storage, submit_job

# Hydra overrides from the command line, so run URIs need no commit
overrides = " ".join(shlex.quote(arg) for arg in sys.argv[1:])


submit_job(
    job_name="ri-scale-survival-model-train",
    username="pekarj",
    cpu=8,
    memory="16Gi",
    gpu=None,
    public=False,
    script=[
        "git clone --single-branch --branch feat/survival-model https://github.com/RationAI/ri-scale.git workdir",
        "cd workdir/survival_model",
        "uv sync --frozen",
        "export MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow",
        f"uv run -m survival_model +experiment=mmci {overrides}",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
