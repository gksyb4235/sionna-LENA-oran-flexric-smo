"""FastAPI boundary for the KPI Advisor Agent."""

from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from agent import AdvisorNotReadyError, get_model_health, run_advisor
from runtime_registration import register_from_environment
from schemas import (
    HealthResponse,
    InvokeRequest,
    InvokeResponse,
    invalid_field_names,
    validate_response_for_intent,
)

INVOKE_TIMEOUT_SECONDS = 60.0

app = FastAPI(
    title="KPI Advisor Agent",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.on_event("startup")
def register_planning_manifest() -> None:
    """Expose the advisor to Planning Agent only after a successful DB upsert/read-back."""
    try:
        app.state.planner_manifest = register_from_environment()
        app.state.planner_registration_error = None
    except Exception as exc:  # noqa: BLE001 - registration failure must not expose a broken candidate.
        app.state.planner_manifest = None
        app.state.planner_registration_error = f"{type(exc).__name__}: {exc}"


def _error(status_code: int, code: str, message: str, **details: object) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error_code": code, "message": message, "details": details},
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error(_request: object, exc: RequestValidationError) -> JSONResponse:
    return _error(
        422,
        "invalid_request",
        "KPI Advisor input validation failed",
        invalid_fields=invalid_field_names(exc.errors()),
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return get_model_health()


@app.post("/invoke", response_model=InvokeResponse)
async def invoke(request: InvokeRequest) -> InvokeResponse | JSONResponse:
    try:
        raw_response = await asyncio.wait_for(run_advisor(request), timeout=INVOKE_TIMEOUT_SECONDS)
        response = InvokeResponse.model_validate(raw_response)
        validate_response_for_intent(response, request.intent)
        return response
    except TimeoutError:
        return _error(504, "advisor_timeout", "KPI Advisor did not respond within 60 seconds")
    except AdvisorNotReadyError as exc:
        return _error(503, "advisor_not_ready", str(exc))
    except ValidationError as exc:
        return _error(
            500,
            "invalid_advisor_response",
            "KPI Advisor produced an invalid response",
            invalid_fields=invalid_field_names(exc.errors()),
        )
    except ValueError as exc:
        return _error(500, "invalid_advisor_response", str(exc))
    except Exception as exc:  # noqa: BLE001 - preserve an atomic public error contract.
        response = getattr(exc, "response", None)
        if response is not None and hasattr(response, "model_dump"):
            return JSONResponse(status_code=502, content=response.model_dump(mode="json"))
        return _error(502, "advisor_error", str(exc))
