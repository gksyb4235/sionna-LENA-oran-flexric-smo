"""GNN_MCP_Server: read-only MCP tools exposing Inference_Service (design.md 7, Requirement 6).

Follows the ``smo/agentic_ai/agents/probe/mcp_server.py`` FastMCP convention
(``from mcp.server.fastmcp import FastMCP``, ``@mcp.tool()``). This server
wraps the real Inference_Service HTTP API (``smo/aimlfw/inference_service``,
default port 8105) and exposes exactly three read-only tools:

- ``predict_batch``  -> ``POST /predict/batch`` (6.1)
- ``marginal_effect`` -> derived from repeated ``POST /predict/batch`` calls (6.2)
- ``current_model``   -> ``GET /status`` (6.3)

Design decisions worth documenting explicitly:

- **No mutation tools** (6.6, 6.7): every tool here only reads
  Inference_Service state. RAN parameter changes are requested exclusively
  through Policy_Manager's A1 publish path (task 11), never from this
  module.
- **Tool results, never protocol-level exceptions, for domain errors**: on
  an Inference_Service error (6.5), a pre-validation failure (6.9), or a
  timeout/unreachable Inference_Service (6.10), the tool functions return a
  plain ``{"error_code", "message", "details"}`` dict -- the same shape
  ``smo.aimlfw.common.errors.ErrorResponse`` uses elsewhere in this
  codebase -- rather than raising. This mirrors how design.md describes
  errors being carried "도구 결과로" (as a tool result), and it is what lets
  Requirement 6.4/6.5 pass-through be a literal dict pass-through of
  Inference_Service's own JSON body.
- **Required vs. optional arguments in the MCP tool schema** (6.8): required
  arguments (``parameter_sets``, ``baseline``, ``control_parameters``,
  ``target_kpis``) are declared with no Python default, so FastMCP's
  generated JSON schema lists them under ``required`` and a genuinely
  *missing* argument is rejected by the MCP framework itself before this
  module's code runs (still without ever calling Inference_Service).
  Range/size limits (batch size, Time_Step, control-parameter count) are
  documented on the schema via ``json_schema_extra`` (so ``list_tools()``
  callers can discover them) but are *not* enforced as literal Pydantic
  constraints -- those are checked explicitly in each tool body first, so
  that an out-of-range value (as opposed to a missing one) is rejected with
  this module's own structured validation error rather than a generic
  framework validation error, before any Inference_Service call.
- **HTTP client injection**: ``set_client_factory``/``reset_client_factory``
  let tests replace the real ``httpx.Client`` (e.g. with an ASGI-transport
  client bound to a Inference_Service ``TestClient`` app, or a stub that
  raises to simulate a timeout) without touching module-level globals
  directly.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Annotated, Any

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from smo.aimlfw.common.constants import (
    CONTROL_PARAMETER_NAMES,
    MAX_TIME_STEP,
    MIN_TIME_STEP,
    TARGET_KPIS,
)
from smo.aimlfw.inference_service.schemas import MAX_BATCH_SIZE

from probing import STEP_DIRECTIONS, step_value

DEFAULT_INFERENCE_SERVICE_BASE_URL = "http://127.0.0.1:8105"
REQUEST_TIMEOUT_SECONDS = 30.0  # Requirement 6.10: 30s timeout, no retry.
MIN_CONTROL_PARAMETERS = 1
MAX_CONTROL_PARAMETERS = 5  # Requirement 6.2: 1..5 Control_Parameter names.
MIN_MARGINAL_EFFECT_PERCENT = -100.0
MAX_MARGINAL_EFFECT_PERCENT = 100.0

mcp = FastMCP("gnn-inference")


def _base_url() -> str:
    return os.getenv("INFERENCE_SERVICE_BASE_URL", DEFAULT_INFERENCE_SERVICE_BASE_URL).rstrip("/")


def _default_client_factory() -> httpx.Client:
    return httpx.Client(base_url=_base_url(), timeout=REQUEST_TIMEOUT_SECONDS)


_client_factory: Callable[[], httpx.Client] = _default_client_factory


def set_client_factory(factory: Callable[[], httpx.Client]) -> None:
    """Override the Inference_Service HTTP client factory (test seam)."""
    global _client_factory
    _client_factory = factory


def reset_client_factory() -> None:
    """Restore the default (env-var configured) Inference_Service HTTP client factory."""
    global _client_factory
    _client_factory = _default_client_factory


class _InferenceUnavailable(RuntimeError):
    """Raised internally when Inference_Service times out or is unreachable (6.10)."""


def _validation_error(message: str, violated_arguments: list[str], **extra: Any) -> dict[str, Any]:
    """Build the input-validation tool-result body for Requirement 6.9."""
    return {
        "error_code": "input_validation_error",
        "message": message,
        "details": {"violated_arguments": violated_arguments, **extra},
    }


def _inference_unavailable_error(message: str) -> dict[str, Any]:
    """Build the tool-result body for Requirement 6.10."""
    return {"error_code": "inference_unavailable", "message": message, "details": {}}


def _passthrough_error(body: dict[str, Any]) -> dict[str, Any]:
    """Pass through an Inference_Service error body unchanged (Requirement 6.5)."""
    return {
        "error_code": body.get("error_code", "inference_service_error"),
        "message": body.get("message", "Inference_Service returned an error"),
        "details": body.get("details", {}),
    }


def _request_inference_service(
    method: str, path: str, *, json_body: dict[str, Any] | None = None
) -> tuple[int, dict[str, Any]]:
    """Issue exactly one HTTP call to Inference_Service, no retry (Requirement 6.10).

    Raises ``_InferenceUnavailable`` on timeout or connection failure so callers
    can turn that into the ``inference_unavailable`` tool result without
    mutating any input arguments they already received.
    """
    try:
        with _client_factory() as client:
            response = client.request(method, path, json=json_body, timeout=REQUEST_TIMEOUT_SECONDS)
    except httpx.TimeoutException as exc:
        raise _InferenceUnavailable(
            f"Inference_Service did not respond within {REQUEST_TIMEOUT_SECONDS:.0f} seconds"
        ) from exc
    except httpx.HTTPError as exc:
        raise _InferenceUnavailable(f"Inference_Service is unreachable: {exc}") from exc
    try:
        body = response.json()
    except ValueError as exc:
        raise _InferenceUnavailable("Inference_Service returned a non-JSON response") from exc
    return response.status_code, body


@mcp.tool()
def predict_batch(
    parameter_sets: Annotated[
        list[dict[str, Any]],
        Field(
            description=(
                "1..64 Parameter_Set entries (each {'cells': {cell_id: {5 Control_Parameter "
                "values}}}), required. Predictions are returned in this same order."
            ),
            json_schema_extra={"minItems": 1, "maxItems": MAX_BATCH_SIZE},
        ),
    ],
    time_step: Annotated[
        int | None,
        Field(
            default=None,
            description="Optional Time_Step, integer 0..4. Defaults to 0 when omitted.",
            json_schema_extra={"minimum": MIN_TIME_STEP, "maximum": MAX_TIME_STEP},
        ),
    ] = None,
) -> dict[str, Any]:
    """Batch_Prediction_Request against Inference_Service (Requirement 6.1)."""
    violations: list[str] = []
    if not isinstance(parameter_sets, list) or not (1 <= len(parameter_sets) <= MAX_BATCH_SIZE):
        violations.append("parameter_sets")
    time_step_is_valid_int = isinstance(time_step, int) and not isinstance(time_step, bool)
    if time_step is not None and (
        not time_step_is_valid_int or not (MIN_TIME_STEP <= time_step <= MAX_TIME_STEP)
    ):
        violations.append("time_step")
    if violations:
        return _validation_error(
            "predict_batch received an invalid argument",
            violations,
            parameter_set_count=len(parameter_sets) if isinstance(parameter_sets, list) else None,
            time_step=time_step,
        )

    try:
        status_code, body = _request_inference_service(
            "POST", "/predict/batch", json_body={"parameter_sets": parameter_sets, "time_step": time_step}
        )
    except _InferenceUnavailable as exc:
        return _inference_unavailable_error(str(exc))

    if status_code != 200:
        return _passthrough_error(body)
    return body  # Requirement 6.4: field names/values/order unchanged, model_name/version included.


@mcp.tool()
def current_model() -> dict[str, Any]:
    """Return Inference_Service's currently loaded/serving model name and version (Requirement 6.3)."""
    try:
        status_code, body = _request_inference_service("GET", "/status")
    except _InferenceUnavailable as exc:
        return _inference_unavailable_error(str(exc))

    if status_code != 200:
        return _passthrough_error(body)
    if body.get("model_load_status") != "loaded":
        return {
            "error_code": "model_not_loaded",
            "message": "Inference_Service has no loaded model artifact",
            "details": {"model_load_status": body.get("model_load_status")},
        }
    return {"model_name": body["model_name"], "model_version": body["model_version"]}


def _build_variant_cells(
    baseline_cells: dict[str, dict[str, Any]], parameter_name: str, direction: str
) -> dict[str, dict[str, Any]] | None:
    """Shift one Control_Parameter by one step in every baseline cell simultaneously.

    Returns None if the shift is out of range for *any* cell -- Requirement
    6.2 defines Marginal_Effect per (Control_Parameter, Target_KPI) pair with
    no cell dimension (unlike Requirement 7's per-cell Probe_Variant), so this
    tool treats a Control_Parameter as one system-wide value to probe rather
    than probing each cell independently.
    """
    updated: dict[str, dict[str, Any]] = {}
    for cell_id, cell_parameters in baseline_cells.items():
        if parameter_name not in cell_parameters:
            return None
        new_value = step_value(parameter_name, cell_parameters[parameter_name], direction)
        if new_value is None:
            return None
        updated_cell = dict(cell_parameters)
        updated_cell[parameter_name] = new_value
        updated[cell_id] = updated_cell
    return updated


def _select_variant_cells(
    baseline_cells: dict[str, dict[str, Any]], parameter_name: str
) -> dict[str, dict[str, Any]] | None:
    """Pick the +1-step variant, falling back to -1, or None if neither direction is valid.

    Design decision (documented per the task instructions, since Requirement
    6.2 does not specify a canonical probing direction the way Requirement
    7.2/7.3 do for Probe_Plan's explicit +1-and--1 variants): this tool
    reports a single Marginal_Effect per (Control_Parameter, Target_KPI)
    pair, so it prefers the +1 direction and only falls back to -1 when +1 is
    out of range for at least one baseline cell. If neither direction keeps
    every baseline cell within its allowed range/step, the Control_Parameter
    is reported as unavailable for this baseline (see
    ``unavailable_control_parameters`` in ``marginal_effect``'s return value)
    rather than guessing a boundary-clamped value.
    """
    for direction in STEP_DIRECTIONS:
        variant_cells = _build_variant_cells(baseline_cells, parameter_name, direction)
        if variant_cells is not None:
            return variant_cells
    return None


@mcp.tool()
def marginal_effect(
    baseline: Annotated[
        dict[str, Any],
        Field(description="Baseline_Parameter_Set, {'cells': {cell_id: {5 Control_Parameter values}}} (required)."),
    ],
    control_parameters: Annotated[
        list[str],
        Field(
            description=(
                "1..5 Control_Parameter names to probe "
                f"(one of {list(CONTROL_PARAMETER_NAMES)}), required."
            ),
            json_schema_extra={"minItems": MIN_CONTROL_PARAMETERS, "maxItems": MAX_CONTROL_PARAMETERS},
        ),
    ],
    target_kpis: Annotated[
        list[str],
        Field(
            description=f"1+ Target_KPI names (one of {list(TARGET_KPIS)}), required.",
            json_schema_extra={"minItems": 1},
        ),
    ],
) -> dict[str, Any]:
    """Marginal_Effect of each Control_Parameter on each Target_KPI (Requirement 6.2)."""
    violations: list[str] = []
    if not isinstance(baseline, dict) or not isinstance(baseline.get("cells"), dict) or not baseline.get("cells"):
        violations.append("baseline")
    if (
        not isinstance(control_parameters, list)
        or not (MIN_CONTROL_PARAMETERS <= len(control_parameters) <= MAX_CONTROL_PARAMETERS)
        or len(set(control_parameters)) != len(control_parameters)
        or not set(control_parameters).issubset(CONTROL_PARAMETER_NAMES)
    ):
        violations.append("control_parameters")
    if not isinstance(target_kpis, list) or not target_kpis or not set(target_kpis).issubset(TARGET_KPIS):
        violations.append("target_kpis")
    if violations:
        return _validation_error(
            "marginal_effect received an invalid argument",
            violations,
            control_parameters=control_parameters,
            target_kpis=target_kpis,
        )

    baseline_cells: dict[str, dict[str, Any]] = baseline["cells"]
    variant_by_parameter: dict[str, dict[str, dict[str, Any]]] = {}
    unavailable_parameters: list[str] = []
    for parameter_name in control_parameters:
        variant_cells = _select_variant_cells(baseline_cells, parameter_name)
        if variant_cells is None:
            unavailable_parameters.append(parameter_name)
        else:
            variant_by_parameter[parameter_name] = variant_cells

    ordered_parameters = [name for name in control_parameters if name in variant_by_parameter]
    parameter_sets = [{"cells": baseline_cells}] + [
        {"cells": variant_by_parameter[name]} for name in ordered_parameters
    ]

    try:
        status_code, body = _request_inference_service(
            "POST", "/predict/batch", json_body={"parameter_sets": parameter_sets, "time_step": None}
        )
    except _InferenceUnavailable as exc:
        return _inference_unavailable_error(str(exc))

    if status_code != 200:
        return _passthrough_error(body)

    predictions = body["predictions"]
    baseline_target_kpi = predictions[0]["target_kpi"]
    effects: dict[str, dict[str, float | None]] = {}
    for offset, parameter_name in enumerate(ordered_parameters, start=1):
        variant_target_kpi = predictions[offset]["target_kpi"]
        kpi_effects: dict[str, float | None] = {}
        for target_kpi in target_kpis:
            baseline_value = baseline_target_kpi.get(target_kpi)
            variant_value = variant_target_kpi.get(target_kpi)
            if baseline_value is None or variant_value is None or baseline_value == 0:
                kpi_effects[target_kpi] = None
                continue
            percent = (variant_value - baseline_value) / abs(baseline_value) * 100
            # Requirement 6.2 only specifies the -100.0..100.0 clamp (unlike Requirement
            # 9.1's Conflict_Analyzer, which additionally rounds to one decimal place at
            # its own layer); this tool intentionally does not round so KPI_Advisor_Agent
            # receives the unrounded value if it needs finer precision.
            kpi_effects[target_kpi] = max(MIN_MARGINAL_EFFECT_PERCENT, min(MAX_MARGINAL_EFFECT_PERCENT, percent))
        effects[parameter_name] = kpi_effects
    for parameter_name in unavailable_parameters:
        effects[parameter_name] = {target_kpi: None for target_kpi in target_kpis}

    return {
        "model_name": body["model_name"],
        "model_version": body["model_version"],
        "marginal_effects": effects,
        "unavailable_control_parameters": unavailable_parameters,
    }


if __name__ == "__main__":
    mcp.run()
