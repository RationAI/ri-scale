"""Usage: uv run scripts/dataset_split.py patients_uri=mlflow-artifacts:/<exp>/<run>/artifacts/patients.parquet"""

import shlex
import sys

from kube_jobs import storage, submit_job

# Hydra overrides from the command line, so run URIs need no commit
overrides = " ".join(shlex.quote(arg) for arg in sys.argv[1:])


submit_job(
    job_name="ri-scale-survival-model-dataset-split",
    username="pekarj",
    cpu=1,
    memory="4Gi",
    gpu=None,
    public=False,
    script=[
        "git clone --single-branch --branch feat/survival-model https://github.com/RationAI/ri-scale.git workdir",
        "cd workdir/survival_model",
        "uv sync --frozen",
        "export MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow",
        f"uv run -m preprocessing.dataset_split +data=split/mmci {overrides}",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
