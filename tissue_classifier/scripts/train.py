from kube_jobs import storage, submit_job


submit_job(
    job_name="ri-scale-tissue-classifier-train",
    username="pekarj",
    cpu=10,
    memory="32Gi",
    gpu="A40",
    public=False,
    script=[
        "git clone --single-branch https://github.com/RationAI/ri-scale.git workdir",
        "cd workdir/tissue_classifier",
        "uv sync --frozen",
        "export MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow",
        "uv run -m tissue_classifier +experiment=mmci",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
