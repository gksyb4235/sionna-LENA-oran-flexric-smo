"""FastAPI boundary for the local Data_Extractor (default port 8101)."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from smo.aimlfw.common.errors import ErrorResponse
from smo.aimlfw.feature_store import FeatureStore

from .errors import DataExtractorError
from .pipeline import extract

DEFAULT_PORT = 8101
DEFAULT_FEATURE_STORE_ROOT = Path(__file__).resolve().parents[1] / "feature_store" / "data"


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExtractRequest(ApiModel):
    result_dir: str = Field(min_length=1)
    feature_group: str = Field(min_length=1, max_length=128)


class ExtractResponse(ApiModel):
    records_written: int = Field(ge=0)
    excluded_rows: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


def _status_code(error_code: str) -> int:
    if error_code in ("csv_not_found", "csv_not_readable", "csv_empty"):
        return 404 if error_code == "csv_not_found" else 422
    return 422


def create_app(feature_store: FeatureStore | None = None) -> FastAPI:
    store = feature_store or FeatureStore(os.getenv("FEATURE_STORE_ROOT", str(DEFAULT_FEATURE_STORE_ROOT)))
    application = FastAPI(title="SMO AIMLFW Data Extractor", version="0.1.0")
    application.state.feature_store = store

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ready", "component": "data-extractor"}

    @application.exception_handler(DataExtractorError)
    async def handle_data_extractor_error(_: Request, exc: DataExtractorError) -> JSONResponse:
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

    @application.post("/extract", response_model=ExtractResponse)
    def extract_endpoint(request: ExtractRequest) -> ExtractResponse:
        outcome = extract(request.result_dir, request.feature_group, application.state.feature_store)
        return ExtractResponse(
            records_written=outcome.records_written,
            excluded_rows=outcome.excluded_rows,
            warnings=outcome.warnings,
        )

    return application


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "smo.aimlfw.data_extractor.server:app",
        host=os.getenv("DATA_EXTRACTOR_HOST", "127.0.0.1"),
        port=int(os.getenv("DATA_EXTRACTOR_PORT", str(DEFAULT_PORT))),
        reload=False,
    )


if __name__ == "__main__":
    main()
