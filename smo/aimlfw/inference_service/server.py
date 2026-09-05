"""FastAPI boundary for the Inference Service (default port 8105)."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pymongo import MongoClient

from smo.aimlfw.common.errors import ErrorResponse
from smo.aimlfw.model_registry import ModelRegistry
from smo.aimlfw.model_storage import ModelStorage

from .engine import InferenceEngine
from .errors import InferenceServiceError
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionResult,
    ServingVersionRequest,
    StatusResponse,
)

DEFAULT_PORT = 8105
DEFAULT_MONGODB_URI = "mongodb://127.0.0.1:27017"
DEFAULT_DATABASE = "aimlfw"
DEFAULT_COLLECTION = "model_registry"
DEFAULT_MODEL_NAME = "gnn"
DEFAULT_STORAGE_ROOT = Path(__file__).resolve().parents[1] / "models"


def _status_code(error_code: str) -> int:
    if error_code == "model_not_loaded":
        return 503
    return 422


def create_app(engine: InferenceEngine | None = None) -> FastAPI:
    if engine is None:
        client = MongoClient(
            os.getenv("MODEL_REGISTRY_MONGODB_URI", DEFAULT_MONGODB_URI),
            serverSelectionTimeoutMS=int(os.getenv("MODEL_REGISTRY_MONGODB_TIMEOUT_MS", "5000")),
        )
        collection = client[
            os.getenv("MODEL_REGISTRY_DATABASE", DEFAULT_DATABASE)
        ][os.getenv("MODEL_REGISTRY_COLLECTION", DEFAULT_COLLECTION)]
        model_storage = ModelStorage(os.getenv("MODEL_STORAGE_ROOT", str(DEFAULT_STORAGE_ROOT)))
        requested_version = os.getenv("INFERENCE_SERVICE_MODEL_VERSION")
        engine = InferenceEngine(
            ModelRegistry(collection, model_storage),
            os.getenv("INFERENCE_SERVICE_MODEL_NAME", DEFAULT_MODEL_NAME),
            int(requested_version) if requested_version else None,
            load_timeout_seconds=float(os.getenv("INFERENCE_SERVICE_LOAD_TIMEOUT_SECONDS", "60")),
        )

    application = FastAPI(title="SMO AIMLFW Inference Service", version="0.1.0")
    application.state.inference_engine = engine

    @application.on_event("startup")
    async def load_model_on_startup() -> None:
        # Requirement 5.1: complete (or definitively fail) loading within
        # load_timeout_seconds before the service starts answering requests.
        await engine.load()

    @application.exception_handler(InferenceServiceError)
    async def handle_inference_error(_: Request, exc: InferenceServiceError) -> JSONResponse:
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

    @application.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        # Requirement 5.10 fallback: FastAPI's default behavior for an uncaught
        # exception is a bare 500 with no error_code/model_name/model_version at
        # all, which would violate 5.10. Engine failures are expected to already
        # arrive as InferenceServiceError (see engine.predict_batch), but this
        # handler is the last line of defense so *any* unhandled exception during
        # request handling still returns a structured body with those fields and
        # never a partial ``predictions`` list.
        body = ErrorResponse(
            error_code="internal_error",
            message="Inference_Service encountered an unexpected error",
            details={"model_name": engine.loaded_model_name, "model_version": engine.loaded_model_version},
        )
        return JSONResponse(status_code=500, content=body.model_dump(mode="json"))

    @application.get("/status", response_model=StatusResponse)
    def get_status() -> StatusResponse:
        return StatusResponse(
            model_load_status=engine.model_load_status,  # type: ignore[arg-type]
            model_name=engine.loaded_model_name,
            model_version=engine.loaded_model_version,
        )

    @application.post("/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
        predictions = engine.predict_batch(request.parameter_sets, request.time_step)
        applied_time_step = 0 if request.time_step is None else request.time_step
        return BatchPredictionResponse(
            model_name=engine.loaded_model_name or engine.model_name,
            model_version=engine.loaded_model_version or 0,
            applied_time_step=applied_time_step,
            predictions=[PredictionResult.model_validate(prediction) for prediction in predictions],
        )

    @application.put("/serving-version", response_model=StatusResponse)
    async def set_serving_version(request: ServingVersionRequest) -> StatusResponse:
        await engine.switch_version(request.model_name, request.model_version)
        return StatusResponse(
            model_load_status="loaded",
            model_name=engine.loaded_model_name,
            model_version=engine.loaded_model_version,
        )

    return application


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "smo.aimlfw.inference_service.server:app",
        host=os.getenv("INFERENCE_SERVICE_HOST", "127.0.0.1"),
        port=int(os.getenv("INFERENCE_SERVICE_PORT", str(DEFAULT_PORT))),
        reload=False,
    )


if __name__ == "__main__":
    main()
