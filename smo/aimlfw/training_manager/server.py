"""FastAPI boundary for the Training Manager (default port 8103)."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pymongo import MongoClient

from smo.aimlfw.common.errors import ErrorResponse
from smo.aimlfw.feature_store import FeatureStore
from smo.aimlfw.model_registry import ModelRegistry
from smo.aimlfw.model_storage import ModelStorage

from .errors import TrainingManagerError
from .jobs import JobManager
from .schemas import CreateJobRequest, CreateJobResponse, JobStatusResponse

DEFAULT_PORT = 8103
DEFAULT_MONGODB_URI = "mongodb://127.0.0.1:27017"
DEFAULT_DATABASE = "aimlfw"
DEFAULT_COLLECTION = "model_registry"
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_STORE_ROOT = _PACKAGE_ROOT / "feature_store" / "data"
DEFAULT_MODEL_STORAGE_ROOT = _PACKAGE_ROOT / "models"
DEFAULT_JOBS_DIR = Path(__file__).resolve().parent / "jobs"


def _status_code(error_code: str) -> int:
    if error_code == "job_not_found":
        return 404
    return 422


def create_app(job_manager: JobManager | None = None) -> FastAPI:
    if job_manager is None:
        client = MongoClient(
            os.getenv("MODEL_REGISTRY_MONGODB_URI", DEFAULT_MONGODB_URI),
            serverSelectionTimeoutMS=int(os.getenv("MODEL_REGISTRY_MONGODB_TIMEOUT_MS", "5000")),
        )
        collection = client[
            os.getenv("MODEL_REGISTRY_DATABASE", DEFAULT_DATABASE)
        ][os.getenv("MODEL_REGISTRY_COLLECTION", DEFAULT_COLLECTION)]
        model_storage = ModelStorage(os.getenv("MODEL_STORAGE_ROOT", str(DEFAULT_MODEL_STORAGE_ROOT)))
        job_manager = JobManager(
            FeatureStore(os.getenv("FEATURE_STORE_ROOT", str(DEFAULT_FEATURE_STORE_ROOT))),
            ModelRegistry(collection, model_storage),
            model_storage,
            jobs_dir=os.getenv("TRAINING_MANAGER_JOBS_DIR", str(DEFAULT_JOBS_DIR)),
            min_training_samples=int(os.getenv("TRAINING_MANAGER_MIN_SAMPLES", "200")),
            max_training_seconds=float(os.getenv("TRAINING_MANAGER_MAX_SECONDS", "3600")),
        )

    application = FastAPI(title="SMO AIMLFW Training Manager", version="0.1.0")
    application.state.job_manager = job_manager

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ready", "component": "training-manager"}

    @application.exception_handler(TrainingManagerError)
    async def handle_training_manager_error(_: Request, exc: TrainingManagerError) -> JSONResponse:
        return JSONResponse(
            status_code=_status_code(exc.error_code),
            content=exc.response.model_dump(mode="json"),
        )

    @application.exception_handler(RequestValidationError)
    async def handle_request_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        body = ErrorResponse(
            error_code="invalid_request",
            message="Request validation failed",
            details={"errors": jsonable_encoder(exc.errors())},
        )
        return JSONResponse(status_code=422, content=body.model_dump(mode="json"))

    @application.post("/jobs", response_model=CreateJobResponse)
    async def create_job(request: CreateJobRequest) -> CreateJobResponse:
        job_id = await job_manager.create_job(
            request.feature_group,
            request.model_name,
            request.seed,
            request.metadata,
        )
        return CreateJobResponse(job_id=job_id)

    @application.get("/jobs/{job_id}", response_model=JobStatusResponse)
    def get_job(job_id: str) -> JobStatusResponse:
        job = job_manager.get_job(job_id)
        return JobStatusResponse.model_validate(job.model_dump(mode="python"))

    return application


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "smo.aimlfw.training_manager.server:app",
        host=os.getenv("TRAINING_MANAGER_HOST", "127.0.0.1"),
        port=int(os.getenv("TRAINING_MANAGER_PORT", str(DEFAULT_PORT))),
        reload=False,
    )


if __name__ == "__main__":
    main()
