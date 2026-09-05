"""Strict public contracts for the KPI Advisor Agent."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from smo.aimlfw.common.constants import MAX_TARGET_CELLS, TTT_ALLOWED_MS, XAppObjective
from smo.aimlfw.common.models import ParameterSet

_HANGUL_PATTERN = re.compile(r"[가-힣]")
_SHELL_COMMAND_PATTERN = re.compile(
    r"(?im)(?:^|[\r\n])\s*(?:\$\s+|#!|(?:sudo\s+)?(?:ba|z|c|fi)?sh\b|powershell\b|cmd(?:\.exe)?\b|"
    r"(?:sudo\s+)?(?:curl|wget|git|docker|podman|systemctl|service|rm|cp|mv|chmod|chown|python\d*|pip\d*|npm|yarn)\b)"
    r"|(?:&&|\|\||`|\$\()"
)
_KUBECTL_PATTERN = re.compile(r"(?i)\bkubectl\b")


class ContractModel(BaseModel):
    """Strict immutable base for all public KPI Advisor models."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PartialCellParameters(ContractModel):
    """One cell's partial xApp parameter overrides."""

    tx_power_dbm: float | None = Field(default=None, ge=30, le=46, multiple_of=1)
    ret_tilt_deg: float | None = Field(default=None, ge=0, le=15, multiple_of=1)
    cio_bias_db: float | None = Field(default=None, ge=-6, le=6, multiple_of=0.5)
    hysteresis_db: float | None = Field(default=None, ge=0, le=10, multiple_of=0.5)
    ttt_ms: int | None = None

    @field_validator("ttt_ms")
    @classmethod
    def validate_ttt(cls, value: int | None) -> int | None:
        if value is not None and value not in TTT_ALLOWED_MS:
            raise ValueError(f"ttt_ms must be one of {list(TTT_ALLOWED_MS)}")
        return value

    @model_validator(mode="after")
    def require_override(self) -> PartialCellParameters:
        if all(getattr(self, name) is None for name in self.__class__.model_fields):
            raise ValueError("at least one Control_Parameter override is required")
        return self


class XAppParameterRequest(ContractModel):
    """Partial cell-specific request completed from recent Feature Records later."""

    target_cells: list[str] = Field(min_length=1, max_length=MAX_TARGET_CELLS)
    parameter_overrides: dict[str, PartialCellParameters] = Field(min_length=1, max_length=MAX_TARGET_CELLS)
    xapp_objective: XAppObjective | None = None


    @field_validator("target_cells")
    @classmethod
    def validate_target_cells(cls, cells: list[str]) -> list[str]:
        normalized = [cell.strip() for cell in cells]
        if any(not cell for cell in normalized):
            raise ValueError("target_cells entries must be non-empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("target_cells entries must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_override_cells(self) -> XAppParameterRequest:
        unknown = sorted(set(self.parameter_overrides).difference(self.target_cells))
        if unknown:
            raise ValueError(f"parameter_overrides contains non-target cells: {unknown}")
        return self


class InvokeRequest(ContractModel):
    """Input accepted by POST /invoke."""

    intent: str = Field(min_length=1, max_length=2000)
    baseline_parameter_set: ParameterSet | None = None
    xapp_request: XAppParameterRequest | None = None
    time_step: int = Field(default=0, ge=0, le=4)

    @field_validator("intent")
    @classmethod
    def normalize_intent(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("intent must not be blank")
        return normalized

    @model_validator(mode="after")
    def require_one_parameter_source(self) -> InvokeRequest:
        supplied = int(self.baseline_parameter_set is not None) + int(self.xapp_request is not None)
        if supplied != 1:
            raise ValueError("exactly one of baseline_parameter_set or xapp_request is required")
        return self


class Recommendation(ContractModel):
    variant_id: str = Field(min_length=1)
    parameter_set: ParameterSet
    improvement_percent: float | None = None


class HealthResponse(ContractModel):
    status: Literal["ready", "not_ready"]
    model_name: str | None = Field(default=None, min_length=1, max_length=128)
    model_version: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_model_state(self) -> HealthResponse:
        has_name = self.model_name is not None
        has_version = self.model_version is not None
        if has_name != has_version:
            raise ValueError("model_name and model_version must be set together")
        if self.status == "ready" and not has_name:
            raise ValueError("ready status requires a loaded model name and version")
        if self.status == "not_ready" and has_name:
            raise ValueError("not_ready status cannot report a loaded model")
        return self


def contains_forbidden_command(value: Any) -> bool:
    """Return whether a nested public value contains a shell or kubectl command."""

    if isinstance(value, BaseModel):
        return contains_forbidden_command(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return any(contains_forbidden_command(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(contains_forbidden_command(item) for item in value)
    if isinstance(value, str):
        return bool(_KUBECTL_PATTERN.search(value) or _SHELL_COMMAND_PATTERN.search(value))
    return False


class InvokeResponse(ContractModel):
    """Atomic successful response returned by POST /invoke."""

    degradation_verdict: Literal["acceptable", "degrading", "unknown"]
    recommendations: list[Recommendation] = Field(default_factory=list, max_length=10)
    evidence_record_id: str = Field(min_length=1)
    rationale_summary: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def forbid_command_strings(self) -> InvokeResponse:
        if contains_forbidden_command(self):
            raise ValueError("KPI Advisor responses must not contain shell or kubectl commands")
        return self


def intent_is_korean(intent: str) -> bool:
    """Return whether the intent contains Korean Hangul text."""

    return bool(_HANGUL_PATTERN.search(intent))


def validate_response_for_intent(response: InvokeResponse, intent: str) -> None:
    """Apply language-dependent constraints that span request and response."""

    if intent_is_korean(intent) and not _HANGUL_PATTERN.search(response.rationale_summary):
        raise ValueError("Korean intent requires a Korean rationale_summary")


def invalid_field_names(errors: list[dict[str, Any]]) -> list[str]:
    """Return stable, unique field paths from Pydantic/FastAPI validation errors."""

    fields: set[str] = set()
    for error in errors:
        location = [str(part) for part in error.get("loc", ()) if part not in {"body", "__root__"}]
        fields.add(".".join(location) if location else "request")
    return sorted(fields)