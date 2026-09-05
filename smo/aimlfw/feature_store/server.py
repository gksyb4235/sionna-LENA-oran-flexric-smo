"""FastAPI boundary for the local Feature Store (default port 8102)."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from smo.aimlfw.common.errors import ErrorResponse

from .errors import FeatureStoreError
from .schemas import (
    ParseRequest,
    ParseResponse,
    RecordsResponse,
    SerializeRequest,
    SerializeResponse,
    UpsertRequest,
    UpsertResponse,
)
from .store import FeatureStore

DEFAULT_PORT = 8102
DEFAULT_STORAGE_ROOT = Path(__file__).resolve().parent / "data"


def _status_code(error_code: str) -> int:
    if error_code == "source_not_found":
        return 404
    if error_code == "file_write_failed":
        return 500
    return 422


def create_app(store: FeatureStore | None = None) -> FastAPI:
    feature_store = store or FeatureStore(os.getenv("FEATURE_STORE_ROOT", str(DEFAULT_STORAGE_ROOT)))
    application = FastAPI(title="SMO AIMLFW Feature Store", version="0.1.0")
    application.state.feature_store = feature_store

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ready", "component": "feature-store"}

    @application.exception_handler(FeatureStoreError)
    async def handle_feature_store_error(_: Request, exc: FeatureStoreError) -> JSONResponse:
        return JSONResponse(
            status_code=_status_code(exc.error_code),
            content=exc.response.model_dump(mode="json"),
        )

    @application.exception_handler(RequestValidationError)
    async def handle_request_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        body = ErrorResponse(
            error_code="invalid_request",
            message="Request validation failed",
            details={"errors": exc.errors()},
        )
        return JSONResponse(status_code=422, content=jsonable_encoder(body))

    @application.post("/records", response_model=UpsertResponse)
    def upsert_records(request: UpsertRequest) -> UpsertResponse:
        upserted, total = feature_store.upsert(request.records)
        return UpsertResponse(upserted=upserted, total=total)

    @application.get("/records", response_model=RecordsResponse)
    def get_records(
        feature_group: str = Query(min_length=1, max_length=128),
        cell_id: str | None = Query(default=None, min_length=1),
        time_step: int | None = Query(default=None, ge=0, le=4),
    ) -> RecordsResponse:
        records = feature_store.records(feature_group, cell_id=cell_id, time_step=time_step)
        return RecordsResponse(records=records, count=len(records))

    @application.post("/serialize", response_model=SerializeResponse)
    def serialize_records(request: SerializeRequest) -> SerializeResponse:
        written = feature_store.serialize(request.feature_group, request.records, request.target_path)
        return SerializeResponse(written=written)

    @application.post("/parse", response_model=ParseResponse)
    def parse_records(request: ParseRequest) -> ParseResponse:
        records = feature_store.parse(request.source_path, request.feature_group)
        return ParseResponse(records=records, count=len(records))

    return application


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "smo.aimlfw.feature_store.server:app",
        host=os.getenv("FEATURE_STORE_HOST", "127.0.0.1"),
        port=int(os.getenv("FEATURE_STORE_PORT", str(DEFAULT_PORT))),
        reload=False,
    )


if __name__ == "__main__":
    main()
