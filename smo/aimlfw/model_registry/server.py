"""FastAPI boundary for the MongoDB Model Registry (default port 8104)."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pymongo import MongoClient

from smo.aimlfw.common.errors import ErrorResponse
from smo.aimlfw.common.models import ModelVersionRecord
from smo.aimlfw.model_storage import ModelStorage

from .errors import ModelRegistryError
from .registry import ModelRegistry
from .schemas import (
    ModelVersionListResponse,
    RegisterModelVersionRequest,
    RegisterModelVersionResponse,
    ServingVersionRequest,
    ServingVersionResponse,
)

DEFAULT_PORT = 8104
DEFAULT_MONGODB_URI = "mongodb://127.0.0.1:27017"
DEFAULT_DATABASE = "aimlfw"
DEFAULT_COLLECTION = "model_registry"
DEFAULT_STORAGE_ROOT = Path(__file__).resolve().parents[1] / "models"


def _status_code(error_code: str) -> int:
    return 404 if error_code == "model_not_found" else 422


def _validation_fields(exc: RequestValidationError) -> list[str]:
    fields = {str(error["loc"][-1]) for error in exc.errors() if error.get("loc")}
    return sorted(fields)


def create_app(registry: ModelRegistry | None = None) -> FastAPI:
    if registry is None:
        client = MongoClient(
            os.getenv("MODEL_REGISTRY_MONGODB_URI", DEFAULT_MONGODB_URI),
            serverSelectionTimeoutMS=int(os.getenv("MODEL_REGISTRY_MONGODB_TIMEOUT_MS", "5000")),
        )
        collection = client[
            os.getenv("MODEL_REGISTRY_DATABASE", DEFAULT_DATABASE)
        ][os.getenv("MODEL_REGISTRY_COLLECTION", DEFAULT_COLLECTION)]
        registry = ModelRegistry(
            collection,
            ModelStorage(os.getenv("MODEL_STORAGE_ROOT", str(DEFAULT_STORAGE_ROOT))),
        )

    application = FastAPI(title="SMO AIMLFW Model Registry", version="0.1.0")
    application.state.model_registry = registry

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ready", "component": "model-registry"}

    @application.exception_handler(ModelRegistryError)
    async def handle_registry_error(_: Request, exc: ModelRegistryError) -> JSONResponse:
        return JSONResponse(
            status_code=_status_code(exc.error_code),
            content=exc.response.model_dump(mode="json"),
        )

    @application.exception_handler(RequestValidationError)
    async def handle_request_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        is_registration = request.method == "POST" and request.url.path.endswith("/versions")
        body = ErrorResponse(
            error_code="invalid_model_metadata" if is_registration else "invalid_request",
            message="Model registration metadata is invalid" if is_registration else "Request validation failed",
            details={
                "violations": _validation_fields(exc),
                "errors": jsonable_encoder(exc.errors()),
            },
        )
        return JSONResponse(status_code=422, content=body.model_dump(mode="json"))

    @application.post(
        "/models/{model_name}/versions",
        response_model=RegisterModelVersionResponse,
    )
    def register_version(
        model_name: str,
        request: RegisterModelVersionRequest,
    ) -> RegisterModelVersionResponse:
        record = registry.register(
            model_name,
            request.feature_group,
            request.metrics,
            request.artifact_uri,
            request.train_split,
            request.validation_split,
            request.train_samples,
            request.validation_samples,
            request.seed,
        )
        return RegisterModelVersionResponse(model_name=record.model_name, version=record.version)

    @application.get(
        "/models/{model_name}/versions",
        response_model=ModelVersionListResponse,
    )
    def list_versions(model_name: str) -> ModelVersionListResponse:
        versions = registry.list_versions(model_name)
        return ModelVersionListResponse(
            model_name=model_name,
            versions=versions,
            count=len(versions),
        )

    @application.get("/models/{model_name}/latest", response_model=ModelVersionRecord)
    def get_latest(model_name: str) -> ModelVersionRecord:
        return registry.latest(model_name)

    @application.get(
        "/models/{model_name}/versions/{version}",
        response_model=ModelVersionRecord,
    )
    def get_version(model_name: str, version: int) -> ModelVersionRecord:
        return registry.get(model_name, version)

    @application.get("/models/{model_name}/serving-version", response_model=ServingVersionResponse)
    def get_serving_version(model_name: str) -> ServingVersionResponse:
        return ServingVersionResponse(model_name=model_name, version=registry.serving_version(model_name))

    @application.put("/models/{model_name}/serving-version", response_model=ServingVersionResponse)
    def set_serving_version(model_name: str, request: ServingVersionRequest) -> ServingVersionResponse:
        registry.mark_serving(model_name, request.version)
        return ServingVersionResponse(model_name=model_name, version=request.version)

    return application


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "smo.aimlfw.model_registry.server:app",
        host=os.getenv("MODEL_REGISTRY_HOST", "127.0.0.1"),
        port=int(os.getenv("MODEL_REGISTRY_PORT", str(DEFAULT_PORT))),
        reload=False,
    )


if __name__ == "__main__":
    main()
