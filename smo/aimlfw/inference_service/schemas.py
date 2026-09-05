"""HTTP request and response contracts for the Inference Service.

``BatchPredictionRequest`` deliberately does *not* reuse the strict
``smo.aimlfw.common.models.ParameterSet``/``CellParameters`` domain models
for its ``parameter_sets`` field. Those models raise on the *first*
out-of-range Control_Parameter value they see, which would turn a batch
containing several violations into a generic FastAPI 422 instead of the
atomic ``parameter_out_of_range`` error Requirement 5.5 requires — one that
must list *every* violating (index, parameter, value) across the whole
batch. Batch-size and Time_Step bounds (5.6, 5.7, 5.12, 5.13) are enforced
by :mod:`.engine` rather than by ``Field`` constraints for the same reason:
a 0-length or 65-length request must reach ``InferenceEngine`` as ordinary
input, not be rejected by Pydantic before the engine can report the
`empty_batch`/`batch_too_large` error code and received count.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_BATCH_SIZE = 64


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CellParametersInput(ApiModel):
    """Unvalidated (range/step) per-cell Control_Parameter values."""

    tx_power_dbm: float
    ret_tilt_deg: float
    cio_bias_db: float
    hysteresis_db: float
    ttt_ms: int


class ParameterSetInput(ApiModel):
    cells: dict[str, CellParametersInput] = Field(min_length=1)


class BatchPredictionRequest(ApiModel):
    """``POST /predict/batch`` request body (design.md Inference_Service, 5.2, 5.4, 5.13)."""

    parameter_sets: list[ParameterSetInput] = Field(default_factory=list)
    time_step: Any = None


class PredictionResult(ApiModel):
    index: int = Field(ge=0)
    target_kpi: dict[str, float]
    cell_kpi: dict[str, dict[str, float]]


class BatchPredictionResponse(ApiModel):
    model_name: str = Field(min_length=1, max_length=128)
    model_version: int = Field(ge=1)
    applied_time_step: int = Field(ge=0, le=4)
    predictions: list[PredictionResult]


class StatusResponse(ApiModel):
    model_load_status: Literal["loaded", "not_loaded"]
    model_name: str | None = None
    model_version: int | None = None


class ServingVersionRequest(ApiModel):
    model_name: str = Field(min_length=1, max_length=128)
    model_version: int = Field(ge=1)


__all__ = [
    "MAX_BATCH_SIZE",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "CellParametersInput",
    "ParameterSetInput",
    "PredictionResult",
    "StatusResponse",
    "ServingVersionRequest",
]
