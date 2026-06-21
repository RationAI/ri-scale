from kube_jobs import storage, submit_job


submit_job(
    job_name="ri-scale-tissue-classifier-dataset-split",
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
        "uv run -m preprocessing.dataset_split +data=split/mmci",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
