"""Usage: uv run scripts/clinical_dataset.py dataset.source.src_dir=/mnt/data/<export folder>"""

import shlex
import sys

from kube_jobs import storage, submit_job

# Hydra overrides from the command line, so run URIs need no commit
overrides = " ".join(shlex.quote(arg) for arg in sys.argv[1:])


submit_job(
    job_name="ri-scale-survival-model-clinical-dataset",
    username="pekarj",
    cpu=1,
    memory="4Gi",
    gpu=None,
    public=False,
    script=[
        "git clone --single-branch --branch feature/survival-prediction https://github.com/RationAI/ri-scale.git workdir",
        "cd workdir/survival_model",
        "uv sync --frozen",
        "export MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow",
        f"uv run -m preprocessing.clinical_dataset +data=raw/mmci {overrides}",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
