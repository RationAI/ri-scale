from kube_jobs import storage, submit_job


submit_job(
    job_name="ri-scale-tissue-classifier-patient-dataset",
    username=...,
    cpu=1,
    memory="2Gi",
    gpu=None,
    public=False,
    script=[
        "git clone --single-branch https://github.com/RationAI/ri-scale.git workdir",
        "cd workdir/tissue_classifier",
        "uv sync --frozen",
        "export MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow",
        "uv run -m preprocessing.patient_dataset +data=raw/mmci",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
