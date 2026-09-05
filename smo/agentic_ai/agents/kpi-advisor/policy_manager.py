"""Policy_Manager: A1_Mediator (port 9000) Policy_Type/Policy_Instance publishing (Requirement 11).

This module is a plain library (no FastAPI routes of its own -- see design.md's
"``Conflict_Analyzer``, ``Temporal_Scheduler``, ``Policy_Manager``는
``KPI_Advisor_Agent``가 임포트하는 라이브러리 모듈") that turns a judged
``ParameterSet``/``TemporalPlan`` into A1 Policy_Type/Policy_Instance calls
against the real ``A1_Mediator_standalone`` FastAPI app
(``A1_Mediator_standalone/A1_Mediator/app/main.py``, default port 9000).

Public surface (design.md's "11. Policy_Manager" section)::

    register_policy_type() -> int
    publish_temporal_plan(plan, approval, ...) -> list[PolicyInstanceResult] | PendingApproval
    publish_parameter_set(ps, target_cells, approval, ...) -> PolicyInstanceResult | PendingApproval
    serialize_instance(ps: ParameterSet) -> dict
    parse_instance(data: dict) -> ParameterSet

Deviation from design.md's literal pseudocode signatures, documented up front:
``publish_temporal_plan``/``publish_parameter_set`` accept two additional
keyword-only arguments, ``degradation_verdict`` and ``violated_threshold_kpis``.
design.md's pseudocode signatures only list ``approval``, but Requirement
11.10's gate ("Degradation_Verdict가 degrading/unknown이고 승인 기록이
없으면 pending_approval") requires *some* verdict to act on, and neither
``ParameterSet`` nor ``TemporalPlan`` (smo.aimlfw.common.models) carries a
Degradation_Verdict field of its own -- that judgment is produced upstream by
``agent.judge_degradation_verdict`` (Requirement 8) and must be handed to
Policy_Manager alongside the approval record it gates. Both keyword arguments
default to ``"acceptable"``/``None`` so a caller that already knows publishing
is approved (e.g. a caller with an existing approval record covering an
``acceptable`` verdict) may omit them.

Key design decisions (mirrors this package's existing conventions):

- **HTTP client injection** (mirrors ``mcp_server.py``'s
  ``set_client_factory``/``reset_client_factory`` seam exactly): tests point
  Policy_Manager at a real in-process A1_Mediator ASGI app instead of a real
  TCP port, and production reads the target from the ``A1_MEDIATOR_BASE_URL``
  environment variable (default ``http://127.0.0.1:9000``).
- **Stable error shape** (mirrors ``agent.py``'s ``ProbeExecutionError``):
  ``PolicyManagerError`` carries an ``ErrorResponse``-shaped
  ``error_code``/``message``/``details`` body for every failure path
  (``roundtrip_violation``, ``policy_type_not_registered``,
  ``parameter_out_of_range``, ``a1_unreachable``, and this module's own
  ``a1_http_error``/``endpoint_not_whitelisted`` for the two failure shapes
  design.md's own error catalog does not name individually).
- **``pending_approval`` is a state, not an error** (design.md's
  cross-cutting decision #4): ``PendingApproval`` is returned as a normal
  value from ``publish_parameter_set``/``publish_temporal_plan``, never
  raised.
- **8-endpoint whitelist as an enforced invariant** (Requirement 11.16): every
  A1_Mediator call in this module funnels through ``_request``, which checks
  ``(method, path_template)`` against ``ENDPOINT_WHITELIST`` *before* issuing
  any HTTP call and raises ``PolicyManagerError("endpoint_not_whitelisted")``
  otherwise. See ``ENDPOINT_WHITELIST``'s docstring for the enumeration and
  which 5 of the 8 the normal publish path actually uses.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator
from smo.aimlfw.common.constants import (
    CONTROL_PARAMETER_NAMES,
    CONTROL_PARAMETER_RANGES,
    TTT_ALLOWED_MS,
    DegradationVerdict,
)
from smo.aimlfw.common.errors import ErrorResponse
from smo.aimlfw.common.models import CellParameters, ParameterSet, TemporalPlan

DEFAULT_A1_MEDIATOR_BASE_URL = "http://127.0.0.1:9000"
CONNECT_TIMEOUT_SECONDS = 10.0  # Requirement 11.15: connect/response timeout before a retry.
STATUS_TIMEOUT_SECONDS = 5.0  # Requirement 11.11: status query must complete within 5 seconds.
MAX_CONNECT_ATTEMPTS = 3  # Requirement 11.15: 1 initial attempt + up to 2 retries.

# Requirement 11.1/design.md "3개 셀 고정": the scenario's 3 fixed cells, in a
# stable order so create_schema.properties/field ordering is deterministic.
FIXED_CELLS: tuple[str, str, str] = ("gNB_5G", "gNB_4G_1", "gNB_4G_2")

# design.md example policy_type_id ("예: 20100 (기존 10000/20008과 충돌 방지)").
DEFAULT_POLICY_TYPE_ID = 20100

# Requirement 11.1: CIO/HYS are 0.5-step decimal parameters that the real
# A1_Mediator's CreateSchema.validate_properties cannot represent (it only
# accepts "integer"/"bool" property types -- see module docstring and
# A1_Mediator_standalone/A1_Mediator/app/main.py). Both are carried as
# 10x-scaled integers instead (e.g. cio_bias_db=0.5 -> ..._x10=5).
_SCALED_PARAMETERS: dict[str, int] = {"cio_bias_db": 10, "hysteresis_db": 10}


def _field_name(cell_id: str, parameter_name: str) -> str:
    suffix = "_x10" if parameter_name in _SCALED_PARAMETERS else ""
    return f"{cell_id}_{parameter_name}{suffix}"


# Requirement 11.1: 3 fixed cells x 5 Control_Parameters = 15 flat integer
# fields. Maps each A1 field name back to (cell_id, parameter_name, scale) so
# serialize_instance/parse_instance/create_schema generation all derive from
# this single source instead of duplicating the naming rule.
FIELD_MAP: dict[str, tuple[str, str, int]] = {
    _field_name(cell_id, parameter_name): (cell_id, parameter_name, _SCALED_PARAMETERS.get(parameter_name, 1))
    for cell_id in FIXED_CELLS
    for parameter_name in CONTROL_PARAMETER_NAMES
}


# ---------------------------------------------------------------------------
# ApprovalRecord: not already defined anywhere in smo.aimlfw.common.models,
# so it is defined here per the task instructions.
# ---------------------------------------------------------------------------


class ApprovalRecord(BaseModel):
    """Explicit human approval for a Temporal_Plan/Parameter_Set publish (Requirement 11.1, 11.10)."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    approver_id: str = Field(min_length=1)
    approved_at: datetime

    @field_validator("approved_at")
    @classmethod
    def validate_approved_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("approved_at must include a UTC offset")
        return value


# ---------------------------------------------------------------------------
# Result / error / attempt-log types.
# ---------------------------------------------------------------------------


class PolicyManagerError(Exception):
    """A stable, ``ErrorResponse``-shaped Policy_Manager failure (mirrors ``agent.ProbeExecutionError``)."""

    def __init__(self, error_code: str, message: str, details: dict[str, Any] | None = None):
        self.response = ErrorResponse(error_code=error_code, message=message, details=details or {})
        super().__init__(message)

    @property
    def error_code(self) -> str:
        return self.response.error_code

    @property
    def details(self) -> dict[str, Any]:
        return self.response.details


@dataclass(frozen=True)
class PolicyInstanceResult:
    """One Policy_Instance's publish outcome (design.md's ``PolicyInstanceResult``)."""

    policy_type_id: int
    policy_instance_id: str
    status: str  # "enforced" on a successful status query, else "status_unavailable" (11.12).


@dataclass(frozen=True)
class PendingApproval:
    """Requirement 11.10's non-error "no Policy_Instance created" state.

    Returned (never raised) by ``publish_parameter_set``/``publish_temporal_plan``
    whenever the Degradation_Verdict is ``degrading``/``unknown`` and no
    ``ApprovalRecord`` was supplied.
    """

    status: Literal["pending_approval"] = "pending_approval"
    degradation_verdict: DegradationVerdict = "unknown"
    violated_threshold_kpis: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PublishAttempt:
    """Requirement 11.9's "호출한 엔드포인트 이름과 시도 시각을 발행 시도 기록에 남긴다" log entry."""

    method: str
    endpoint: str
    attempted_at: datetime
    status_code: int | None
    outcome: Literal["success", "http_error", "connection_error"]


_publish_attempts: list[PublishAttempt] = []


def get_publish_attempts() -> list[PublishAttempt]:
    """Return a snapshot of every recorded publish attempt (Requirement 11.9)."""
    return list(_publish_attempts)


def clear_publish_attempts() -> None:
    """Reset the publish-attempt log (test seam)."""
    _publish_attempts.clear()


def _record_attempt(
    method: str,
    endpoint: str,
    status_code: int | None,
    outcome: Literal["success", "http_error", "connection_error"],
) -> None:
    _publish_attempts.append(
        PublishAttempt(
            method=method,
            endpoint=endpoint,
            attempted_at=datetime.now(UTC),
            status_code=status_code,
            outcome=outcome,
        )
    )


# ---------------------------------------------------------------------------
# HTTP client injection (mirrors mcp_server.py's set_client_factory/
# reset_client_factory pattern exactly).
# ---------------------------------------------------------------------------


def _base_url() -> str:
    return os.getenv("A1_MEDIATOR_BASE_URL", DEFAULT_A1_MEDIATOR_BASE_URL).rstrip("/")


def _default_client_factory() -> httpx.Client:
    return httpx.Client(base_url=_base_url())


_client_factory: Callable[[], httpx.Client] = _default_client_factory


def set_client_factory(factory: Callable[[], httpx.Client]) -> None:
    """Override the A1_Mediator HTTP client factory (test seam)."""
    global _client_factory
    _client_factory = factory


def reset_client_factory() -> None:
    """Restore the default (env-var configured) A1_Mediator HTTP client factory."""
    global _client_factory
    _client_factory = _default_client_factory


# ---------------------------------------------------------------------------
# Requirement 11.16: the 8-endpoint whitelist, enforced (not just habitual).
# ---------------------------------------------------------------------------

# The real A1_Mediator_standalone app (app/main.py) exposes exactly 9 routes:
# healthcheck + these 8 Policy_Type/Policy_Instance routes. Requirement
# 11.16 says Policy_Manager "SHALL 포트 9000의 A1_Mediator가 이미 제공하는
# 8개 Policy_Type/Policy_Instance 엔드포인트만 호출하고 그 외 엔드포인트를
# 호출하지 않는다" -- i.e. it may call any of these 8 (never anything else,
# and never GET /a1-p/healthcheck through this whitelist path; healthcheck is
# handled separately by ``healthcheck()`` below, per design.md's "GET
# /a1-p/healthcheck는 연결성 확인에만 사용한다"). Per design.md's own
# enumeration, the *normal* publish path (register_policy_type,
# publish_parameter_set, publish_temporal_plan) only ever exercises 1, 2, 3,
# 5, 6 below; 4, 7, 8 are implemented as plain convenience functions for
# operational use (inspecting/cleaning up A1_Mediator state) but are never
# called from the publish path itself.
#   1. GET    /a1-p/policytypes                                              -> list_policy_type_ids()
#   2. GET    /a1-p/policytypes/{policy_type_id}                            -> get_policy_type()
#   3. PUT    /a1-p/policytypes/{policy_type_id}                            -> register_policy_type()
#   4. DELETE /a1-p/policytypes/{policy_type_id} -> delete_policy_type()
#   5. PUT /a1-p/policytypes/{policy_type_id}/policies/{instance_id} -> publish functions
#   6. GET /a1-p/policytypes/{policy_type_id}/policies/{instance_id}/status -> status
#   7. GET /a1-p/policytypes/{policy_type_id}/policies -> list_policy_instances()
#   8. DELETE /a1-p/policytypes/{policy_type_id}/policies/{instance_id} -> delete_policy_instance()
ENDPOINT_WHITELIST: frozenset[tuple[str, str]] = frozenset(
    {
        ("GET", "/a1-p/policytypes"),
        ("GET", "/a1-p/policytypes/{policy_type_id}"),
        ("PUT", "/a1-p/policytypes/{policy_type_id}"),
        ("DELETE", "/a1-p/policytypes/{policy_type_id}"),
        ("PUT", "/a1-p/policytypes/{policy_type_id}/policies/{policy_instance_id}"),
        ("GET", "/a1-p/policytypes/{policy_type_id}/policies/{policy_instance_id}/status"),
        ("GET", "/a1-p/policytypes/{policy_type_id}/policies"),
        ("DELETE", "/a1-p/policytypes/{policy_type_id}/policies/{policy_instance_id}"),
    }
)

_HEALTHCHECK_ENDPOINT: tuple[str, str] = ("GET", "/a1-p/healthcheck")


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


def _request(
    method: str,
    path_template: str,
    path_params: dict[str, Any],
    *,
    json_body: Any = None,
    timeout: float = CONNECT_TIMEOUT_SECONDS,
    max_attempts: int = MAX_CONNECT_ATTEMPTS,
) -> httpx.Response:
    """Issue exactly one whitelisted A1_Mediator call, retrying only on connection failure/timeout.

    Requirement 11.16: raises ``PolicyManagerError("endpoint_not_whitelisted")``
    immediately (no HTTP call at all) for any ``(method, path_template)`` not
    in ``ENDPOINT_WHITELIST``.

    Requirement 11.15: connection failure or a response that does not arrive
    within ``timeout`` seconds is retried up to ``max_attempts - 1`` more
    times (default 3 total attempts); if every attempt fails this way, raises
    ``PolicyManagerError("a1_unreachable")``. Callers that must not retry (the
    single Requirement 11.11 status query) pass ``max_attempts=1``.

    Requirement 11.9: any HTTP response (including 4xx/5xx) is returned
    as-is, never retried here -- callers are responsible for turning a 4xx/5xx
    status code into their own ``a1_http_error`` failure, since the right
    error_code/handling differs by call site (e.g. idempotent registration
    vs. instance creation).
    """
    if (method, path_template) not in ENDPOINT_WHITELIST:
        raise PolicyManagerError(
            "endpoint_not_whitelisted",
            f"{method} {path_template} is not one of the 8 whitelisted A1_Mediator endpoints",
            {"method": method, "path_template": path_template},
        )
    path = path_template.format(**path_params)
    last_reason: str | None = None
    for _attempt in range(max_attempts):
        try:
            with _client_factory() as client:
                return client.request(method, path, json=json_body, timeout=timeout)
        except httpx.TimeoutException as exc:
            last_reason = f"A1_Mediator did not respond within {timeout:.0f} seconds: {exc}"
        except httpx.HTTPError as exc:
            last_reason = f"A1_Mediator is unreachable: {exc}"
        if method == "PUT":
            _record_attempt(method, path, None, "connection_error")
    raise PolicyManagerError(
        "a1_unreachable",
        f"A1_Mediator {method} {path} failed after {max_attempts} attempt(s)",
        {"method": method, "path": path, "attempts": max_attempts, "reason": last_reason},
    )


def healthcheck() -> bool:
    """Connectivity-only check via ``GET /a1-p/healthcheck`` (outside the 11.16 whitelist by design)."""
    method, path = _HEALTHCHECK_ENDPOINT
    try:
        with _client_factory() as client:
            response = client.request(method, path, timeout=CONNECT_TIMEOUT_SECONDS)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


# ---------------------------------------------------------------------------
# Requirement 11.14: Control_Parameter validation, reusing the single-sourced
# constants agent.py's ParameterSet construction also relies on.
# ---------------------------------------------------------------------------


def _validate_cell_parameters(cell_id: str, parameters: CellParameters) -> list[dict[str, Any]]:
    """Return every Requirement 11.14 violation for one cell's 5 Control_Parameters.

    ``CellParameters`` (smo.aimlfw.common.models) already enforces every
    range/step/allowed-value rule at construction time via strict Pydantic
    validation, so in practice this only finds a violation if a caller built
    one bypassing that validation (e.g. ``CellParameters.model_construct``).
    It exists to make Requirement 11.14 an explicit, independently-checkable
    invariant at the Policy_Manager boundary rather than relying solely on
    ParameterSet's own constructor validation.
    """
    violations: list[dict[str, Any]] = []
    for parameter_name in CONTROL_PARAMETER_NAMES:
        value = getattr(parameters, parameter_name, None)
        if value is None:
            violations.append({"cell_id": cell_id, "parameter_name": parameter_name, "value": None})
            continue
        if parameter_name == "ttt_ms":
            if value not in TTT_ALLOWED_MS:
                violations.append({"cell_id": cell_id, "parameter_name": parameter_name, "value": value})
            continue
        rule = CONTROL_PARAMETER_RANGES[parameter_name]
        minimum, maximum, step = Decimal(str(rule["min"])), Decimal(str(rule["max"])), Decimal(str(rule["step"]))
        decimal_value = Decimal(str(value))
        if not (minimum <= decimal_value <= maximum) or (decimal_value - minimum) % step != 0:
            violations.append({"cell_id": cell_id, "parameter_name": parameter_name, "value": value})
    return violations


def _validate_complete_policy_parameter_set(ps: ParameterSet) -> list[dict[str, Any]]:
    """Validate the exact three-cell shape required by the A1 flat schema."""
    violations: list[dict[str, Any]] = []
    for cell_id in FIXED_CELLS:
        parameters = ps.cells.get(cell_id)
        if parameters is None:
            violations.append(
                {
                    "cell_id": cell_id,
                    "parameter_name": None,
                    "value": None,
                    "reason": "missing_fixed_cell",
                }
            )
            continue
        violations.extend(_validate_cell_parameters(cell_id, parameters))
    for cell_id in sorted(set(ps.cells) - set(FIXED_CELLS)):
        violations.append(
            {
                "cell_id": cell_id,
                "parameter_name": None,
                "value": None,
                "reason": "unsupported_cell",
            }
        )
    return violations


# ---------------------------------------------------------------------------
# Requirement 11.5, 11.6, 11.7, 11.8: serialize/parse round-trip.
# ---------------------------------------------------------------------------


def serialize_instance(ps: ParameterSet) -> dict[str, int]:
    """Flatten a ParameterSet into its A1 Policy_Instance body (Requirement 11.5).

    Only fields for cells that are among ``FIXED_CELLS`` are included -- the
    Policy_Type's ``create_schema`` is fixed to the 3 scenario cells, so a
    cell outside that set has no corresponding field and is silently dropped
    here. That drop is exactly what makes ``_check_roundtrip`` detect (and
    reject via ``roundtrip_violation``, 11.7/11.8) a Parameter_Set whose
    cells are not a subset of ``FIXED_CELLS``.
    """
    data: dict[str, int] = {}
    for cell_id, parameters in ps.cells.items():
        if cell_id not in FIXED_CELLS:
            continue
        for parameter_name in CONTROL_PARAMETER_NAMES:
            value = getattr(parameters, parameter_name)
            scale = _SCALED_PARAMETERS.get(parameter_name, 1)
            data[_field_name(cell_id, parameter_name)] = round(value * scale)
    return data


def parse_instance(data: dict[str, Any]) -> ParameterSet:
    """Reconstruct a ParameterSet from a flat A1 Policy_Instance body (Requirement 11.6)."""
    cells_raw: dict[str, dict[str, float | int]] = {}
    for field_name, value in data.items():
        mapping = FIELD_MAP.get(field_name)
        if mapping is None:
            continue
        cell_id, parameter_name, scale = mapping
        raw_value: float | int = value / scale if scale != 1 else value
        cells_raw.setdefault(cell_id, {})[parameter_name] = raw_value
    cells = {cell_id: CellParameters.model_validate(values) for cell_id, values in cells_raw.items()}
    return ParameterSet(cells=cells)


def _check_roundtrip(ps: ParameterSet) -> None:
    """Requirement 11.7/11.8: serialize -> parse must reproduce ``ps`` exactly (cell set + all 5 values per cell)."""
    serialized = serialize_instance(ps)
    try:
        parsed = parse_instance(serialized)
    except Exception as exc:  # e.g. a pydantic ValidationError from a malformed round-trip.
        raise PolicyManagerError(
            "roundtrip_violation",
            "Policy_Instance serialization round-trip raised while parsing",
            {"reason": str(exc)},
        ) from exc
    if set(parsed.cells) != set(ps.cells) or any(parsed.cells[cell_id] != ps.cells[cell_id] for cell_id in ps.cells):
        raise PolicyManagerError(
            "roundtrip_violation",
            "Policy_Instance serialization round-trip did not reproduce the original Parameter_Set",
            {"expected_cells": sorted(ps.cells), "actual_cells": sorted(parsed.cells)},
        )


# ---------------------------------------------------------------------------
# Requirement 11.1, 11.2: Policy_Type registration.
# ---------------------------------------------------------------------------


def _create_schema_properties() -> dict[str, dict[str, Any]]:
    """Build the 15 bounded flat integer fields required by Requirement 11.1.

    The real A1_Mediator's ``CreateSchema.validate_properties`` only accepts
    ``"integer"``/``"bool"`` property types (see module docstring); every one
    of this Policy_Type's 15 fields is an integer (TxP/RET/TTT natively, CIO/
    HYS via the ``_x10`` scaling in ``FIELD_MAP``). Bounds and discrete TTT
    values are included so the published policy contract is self-describing.
    """
    properties: dict[str, dict[str, Any]] = {}
    for field_name, (_cell_id, parameter_name, scale) in FIELD_MAP.items():
        if parameter_name == "ttt_ms":
            properties[field_name] = {"type": "integer", "enum": list(TTT_ALLOWED_MS)}
            continue
        rule = CONTROL_PARAMETER_RANGES[parameter_name]
        properties[field_name] = {
            "type": "integer",
            "minimum": round(float(rule["min"]) * scale),
            "maximum": round(float(rule["max"]) * scale),
            "multipleOf": round(float(rule["step"]) * scale),
        }
    return properties


def list_policy_type_ids() -> list[int]:
    """``GET /a1-p/policytypes`` (whitelist entry 1)."""
    response = _request("GET", "/a1-p/policytypes", {})
    if response.status_code != 200:
        raise PolicyManagerError(
            "a1_http_error",
            f"A1_Mediator returned status {response.status_code} listing Policy_Types",
            {"status_code": response.status_code, "response_body": _safe_json(response)},
        )
    return list(response.json())


def get_policy_type(policy_type_id: int) -> dict[str, Any] | None:
    """``GET /a1-p/policytypes/{policy_type_id}`` (whitelist entry 2). ``None`` if not registered (404)."""
    response = _request("GET", "/a1-p/policytypes/{policy_type_id}", {"policy_type_id": policy_type_id})
    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise PolicyManagerError(
            "a1_http_error",
            f"A1_Mediator returned status {response.status_code} fetching Policy_Type {policy_type_id}",
            {"status_code": response.status_code, "response_body": _safe_json(response)},
        )
    return response.json()


def register_policy_type(policy_type_id: int = DEFAULT_POLICY_TYPE_ID) -> int:
    """Register the fixed 15-field Policy_Type, or return the existing id without re-registering (11.1, 11.2).

    Checks registration first via ``GET /a1-p/policytypes/{id}`` (whitelist
    entry 2) rather than blindly issuing the ``PUT`` -- the real A1_Mediator
    returns HTTP 400 on a duplicate ``PUT`` (see
    ``A1_Mediator_standalone/A1_Mediator/app/main.py``'s ``create_policy_type``),
    so this idempotency check must happen in Policy_Manager itself.
    """
    if get_policy_type(policy_type_id) is not None:
        return policy_type_id

    body = {
        "name": "kpi_advisor_control_parameters",
        "description": "3-cell x 5-Control_Parameter flat integer schema for KPI_Advisor_Agent RAN control publishing.",
        "policy_type_id": policy_type_id,
        "create_schema": {
            "type": "object",
            "properties": _create_schema_properties(),
            "additionalProperties": False,
        },
    }
    endpoint = f"/a1-p/policytypes/{policy_type_id}"
    response = _request("PUT", "/a1-p/policytypes/{policy_type_id}", {"policy_type_id": policy_type_id}, json_body=body)
    if response.status_code >= 400:
        _record_attempt("PUT", endpoint, response.status_code, "http_error")
        raise PolicyManagerError(
            "a1_http_error",
            f"A1_Mediator rejected Policy_Type registration with status {response.status_code}",
            {"status_code": response.status_code, "response_body": _safe_json(response)},
        )
    _record_attempt("PUT", endpoint, response.status_code, "success")
    return policy_type_id


def _ensure_policy_type_registered(policy_type_id: int) -> None:
    """Requirement 11.13: reject publishing against an unregistered Policy_Type, no instance created."""
    if get_policy_type(policy_type_id) is None:
        raise PolicyManagerError(
            "policy_type_not_registered",
            f"Policy_Type {policy_type_id} is not registered with A1_Mediator",
            {"policy_type_id": policy_type_id},
        )


# ---------------------------------------------------------------------------
# Requirement 11.3, 11.4, 11.9, 11.11, 11.12: Policy_Instance creation + status.
# ---------------------------------------------------------------------------


def _create_and_check_instance(
    policy_type_id: int, instance_id: str, cells: dict[str, CellParameters]
) -> PolicyInstanceResult:
    """``PUT`` a Policy_Instance then perform the single Requirement 11.11 status query."""
    body = serialize_instance(ParameterSet(cells=cells))
    path_params = {"policy_type_id": policy_type_id, "policy_instance_id": instance_id}
    create_endpoint = f"/a1-p/policytypes/{policy_type_id}/policies/{instance_id}"

    create_response = _request(
        "PUT", "/a1-p/policytypes/{policy_type_id}/policies/{policy_instance_id}", path_params, json_body=body
    )
    if create_response.status_code >= 400:
        _record_attempt("PUT", create_endpoint, create_response.status_code, "http_error")
        raise PolicyManagerError(
            "a1_http_error",
            f"A1_Mediator rejected Policy_Instance creation with status {create_response.status_code}",
            {"status_code": create_response.status_code, "response_body": _safe_json(create_response)},
        )
    _record_attempt("PUT", create_endpoint, create_response.status_code, "success")

    # Requirement 11.11/11.12: exactly one status query attempt (max_attempts=1,
    # separate from the 3-attempt connection-retry policy used for the create
    # call), within 5 seconds. Any failure (HTTP error or unreachable) keeps
    # the just-created instance and reports "status_unavailable" -- it never
    # deletes the instance nor propagates as a publish failure.
    status_endpoint = f"{create_endpoint}/status"
    status_value = "status_unavailable"
    try:
        status_response = _request(
            "GET",
            "/a1-p/policytypes/{policy_type_id}/policies/{policy_instance_id}/status",
            path_params,
            timeout=STATUS_TIMEOUT_SECONDS,
            max_attempts=1,
        )
    except PolicyManagerError:
        _record_attempt("GET", status_endpoint, None, "http_error")
    else:
        if status_response.status_code == 200:
            # The real A1_Mediator's status body is just the stored instance
            # data (no discrete status field of its own -- see
            # get_policy_instance_status in app/main.py); a 200 within the
            # time budget is treated as "enforced" per O-RAN A1AP's
            # ENFORCED/NOT_ENFORCED status vocabulary.
            status_value = "enforced"
            _record_attempt("GET", status_endpoint, status_response.status_code, "success")
        else:
            _record_attempt("GET", status_endpoint, status_response.status_code, "http_error")

    return PolicyInstanceResult(policy_type_id=policy_type_id, policy_instance_id=instance_id, status=status_value)


def get_policy_instance_status(policy_type_id: int, policy_instance_id: str) -> dict[str, Any]:
    """``GET .../policies/{id}/status`` (whitelist entry 6), for direct/operational use outside publish."""
    path_params = {"policy_type_id": policy_type_id, "policy_instance_id": policy_instance_id}
    response = _request(
        "GET", "/a1-p/policytypes/{policy_type_id}/policies/{policy_instance_id}/status", path_params
    )
    if response.status_code != 200:
        raise PolicyManagerError(
            "a1_http_error",
            f"A1_Mediator returned status {response.status_code} querying Policy_Instance {policy_instance_id}",
            {"status_code": response.status_code, "response_body": _safe_json(response)},
        )
    return response.json()


def list_policy_instances(policy_type_id: int) -> list[str]:
    """``GET .../policies`` (whitelist entry 7, not used by the publish path)."""
    response = _request("GET", "/a1-p/policytypes/{policy_type_id}/policies", {"policy_type_id": policy_type_id})
    if response.status_code != 200:
        raise PolicyManagerError(
            "a1_http_error",
            f"A1_Mediator returned status {response.status_code} listing Policy_Instances",
            {"status_code": response.status_code, "response_body": _safe_json(response)},
        )
    return list(response.json())


def delete_policy_type(policy_type_id: int) -> None:
    """``DELETE /a1-p/policytypes/{id}`` (whitelist entry 4, not used by the publish path)."""
    response = _request("DELETE", "/a1-p/policytypes/{policy_type_id}", {"policy_type_id": policy_type_id})
    if response.status_code >= 400:
        raise PolicyManagerError(
            "a1_http_error",
            f"A1_Mediator returned status {response.status_code} deleting Policy_Type {policy_type_id}",
            {"status_code": response.status_code, "response_body": _safe_json(response)},
        )


def delete_policy_instance(policy_type_id: int, policy_instance_id: str) -> None:
    """``DELETE .../policies/{id}`` (whitelist entry 8, not used by the publish path)."""
    path_params = {"policy_type_id": policy_type_id, "policy_instance_id": policy_instance_id}
    response = _request(
        "DELETE", "/a1-p/policytypes/{policy_type_id}/policies/{policy_instance_id}", path_params
    )
    if response.status_code >= 400:
        raise PolicyManagerError(
            "a1_http_error",
            f"A1_Mediator returned status {response.status_code} deleting Policy_Instance {policy_instance_id}",
            {"status_code": response.status_code, "response_body": _safe_json(response)},
        )


# ---------------------------------------------------------------------------
# Requirement 11.3, 11.4, 11.10: the two public publish entry points.
# ---------------------------------------------------------------------------


def publish_parameter_set(
    ps: ParameterSet,
    target_cells: list[str],
    approval: ApprovalRecord | None = None,
    *,
    degradation_verdict: DegradationVerdict = "acceptable",
    violated_threshold_kpis: list[str] | None = None,
    policy_type_id: int = DEFAULT_POLICY_TYPE_ID,
) -> PolicyInstanceResult | PendingApproval:
    """Publish one Policy_Instance for the schema's exact three fixed cells (Requirement 11.3).

    Order of checks mirrors design.md's sequence diagram: approval gate
    (11.10) first (no A1_Mediator call at all if it fires) -> parameter
    validation (11.14) -> Policy_Type registration check (11.13) -> roundtrip
    check (11.7/11.8) -> create + status query (11.9, 11.11, 11.12).
    """
    if degradation_verdict in ("degrading", "unknown") and approval is None:
        return PendingApproval(
            degradation_verdict=degradation_verdict,
            violated_threshold_kpis=list(violated_threshold_kpis or []),
        )

    violations: list[dict[str, Any]] = _validate_complete_policy_parameter_set(ps)
    if set(target_cells) != set(FIXED_CELLS) or len(target_cells) != len(FIXED_CELLS):
        violations.append(
            {
                "parameter_name": None,
                "value": list(target_cells),
                "reason": "target_cells_must_match_fixed_policy_cells",
                "expected_cells": list(FIXED_CELLS),
            }
        )
    for cell_id in target_cells:
        if cell_id not in ps.cells:
            violations.append({"cell_id": cell_id, "parameter_name": None, "value": None, "reason": "missing_cell"})
    if violations:
        raise PolicyManagerError(
            "parameter_out_of_range",
            "One or more Control_Parameter values are missing or out of range",
            {"violations": violations},
        )

    _ensure_policy_type_registered(policy_type_id)

    _check_roundtrip(ps)

    instance_id = f"instance-{uuid.uuid4().hex}"
    return _create_and_check_instance(policy_type_id, instance_id, dict(ps.cells))


def publish_temporal_plan(
    plan: TemporalPlan,
    approval: ApprovalRecord | None = None,
    *,
    degradation_verdict: DegradationVerdict = "acceptable",
    violated_threshold_kpis: list[str] | None = None,
    policy_type_id: int = DEFAULT_POLICY_TYPE_ID,
) -> list[PolicyInstanceResult] | PendingApproval:
    """Publish exactly 5 Policy_Instances, one per Time_Step 0..4 (Requirement 11.4).

    ``plan.assignments`` is already guaranteed by ``TemporalPlan``'s own
    Pydantic validator (smo.aimlfw.common.models) to contain exactly 5
    entries with ``time_step`` 0..4 in ascending order, so the returned list
    is exactly 5 long whenever this does not return ``PendingApproval`` or
    raise.
    """
    if degradation_verdict in ("degrading", "unknown") and approval is None:
        return PendingApproval(
            degradation_verdict=degradation_verdict,
            violated_threshold_kpis=list(violated_threshold_kpis or []),
        )

    _ensure_policy_type_registered(policy_type_id)

    results: list[PolicyInstanceResult] = []
    for assignment in plan.assignments:
        ps = assignment.parameter_set
        violations = _validate_complete_policy_parameter_set(ps)
        if violations:
            raise PolicyManagerError(
                "parameter_out_of_range",
                f"Time_Step {assignment.time_step} has a missing or out-of-range Control_Parameter",
                {"time_step": assignment.time_step, "violations": violations},
            )
        _check_roundtrip(ps)
        instance_id = f"instance-ts{assignment.time_step}-{uuid.uuid4().hex}"
        results.append(_create_and_check_instance(policy_type_id, instance_id, dict(ps.cells)))
    return results


__all__ = [
    "ApprovalRecord",
    "DEFAULT_POLICY_TYPE_ID",
    "ENDPOINT_WHITELIST",
    "FIELD_MAP",
    "FIXED_CELLS",
    "PendingApproval",
    "PolicyInstanceResult",
    "PolicyManagerError",
    "PublishAttempt",
    "clear_publish_attempts",
    "delete_policy_instance",
    "delete_policy_type",
    "get_policy_instance_status",
    "get_policy_type",
    "get_publish_attempts",
    "healthcheck",
    "list_policy_instances",
    "list_policy_type_ids",
    "parse_instance",
    "publish_parameter_set",
    "publish_temporal_plan",
    "register_policy_type",
    "reset_client_factory",
    "serialize_instance",
    "set_client_factory",
]
