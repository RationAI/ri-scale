"""Usage: uv run scripts/cohort_analysis.py MMCI=mlflow-artifacts:/<exp>/<run>/artifacts/patients.parquet MUG=mlflow-artifacts:/<exp>/<run>/artifacts/patients.parquet"""

import shlex
import sys

from kube_jobs import submit_job

# NAME=patients.parquet URIs from the command line
cohorts = " ".join(f"--cohort {shlex.quote(arg)}" for arg in sys.argv[1:])


submit_job(
    job_name="ri-scale-survival-model-cohort-analysis",
    username="pekarj",
    cpu=2,
    memory="8Gi",
    gpu=None,
    public=False,
    script=[
        "git clone --single-branch --branch feature/survival-prediction https://github.com/RationAI/ri-scale.git workdir",
        "cd workdir/survival_model",
        "uv sync --frozen",
        "export MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow",
        f"uv run -m survival_model.cohort_analysis {cohorts} --out /tmp/cohort_analysis",
    ],
)
