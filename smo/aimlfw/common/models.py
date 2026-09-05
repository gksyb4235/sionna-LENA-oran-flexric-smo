"""Pydantic domain contracts shared by all AIMLFW components."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .constants import (
    CONTROL_PARAMETER_NAMES,
    CONTROL_PARAMETER_RANGES,
    MAX_SEED,
    MAX_TARGET_CELLS,
    TARGET_KPIS,
    TTT_ALLOWED_MS,
    ControlParameterName,
    DataQuality,
    DegradationVerdict,
    XAppObjective,
)


class DomainModel(BaseModel):
    """Strict, immutable and forward-compatible domain model base."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _finite(value: float | int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number")


def _validate_parameter(name: str, value: float | int) -> None:
    _finite(value, name)
    rule = CONTROL_PARAMETER_RANGES[name]
    if name == "ttt_ms":
        if value not in TTT_ALLOWED_MS:
            raise ValueError(f"ttt_ms must be one of {list(TTT_ALLOWED_MS)}")
        return
    minimum, maximum = Decimal(str(rule["min"])), Decimal(str(rule["max"]))
    decimal_value, step = Decimal(str(value)), Decimal(str(rule["step"]))
    if not minimum <= decimal_value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    if (decimal_value - minimum) % step != 0:
        raise ValueError(f"{name} must align to step {step} from {minimum}")


def _validate_named_finite_map(values: dict[str, float | None], field_name: str) -> dict[str, float | None]:
    if any(not name for name in values):
        raise ValueError(f"{field_name} names must be non-empty")
    for name, value in values.items():
        if value is not None:
            _finite(value, f"{field_name}.{name}")
    return values


def _validate_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a UTC offset")
    if value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{field_name} must be UTC")
    return value


class CellParameters(DomainModel):
    tx_power_dbm: float
    ret_tilt_deg: float
    cio_bias_db: float
    hysteresis_db: float
    ttt_ms: int

    @model_validator(mode="after")
    def validate_ranges(self) -> CellParameters:
        for name in CONTROL_PARAMETER_NAMES:
            _validate_parameter(name, getattr(self, name))
        return self


class ParameterSet(DomainModel):
    """A complete five-parameter value set for every target cell."""

    cells: dict[str, CellParameters]

    @field_validator("cells")
    @classmethod
    def validate_cells(cls, cells: dict[str, CellParameters]) -> dict[str, CellParameters]:
        if not 1 <= len(cells) <= MAX_TARGET_CELLS:
            raise ValueError(f"cells must contain 1..{MAX_TARGET_CELLS} entries")
        if any(not cell_id.strip() for cell_id in cells):
            raise ValueError("cell_id must be non-empty")
        return cells


class SampleCount(DomainModel):
    valid: int = Field(ge=0)
    excluded: int = Field(ge=0)


class FeatureRecord(DomainModel):
    feature_group: str = Field(min_length=1, max_length=128)
    time_step: int = Field(ge=0, le=4)
    cell_id: str = Field(min_length=1)
    control_parameters: CellParameters
    features: dict[str, float | None]
    sample_counts: dict[str, SampleCount]
    data_quality: DataQuality
    source_dir: str = Field(min_length=1)
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)

    @field_validator("features")
    @classmethod
    def validate_features(cls, values: dict[str, float | None]) -> dict[str, float | None]:
        return _validate_named_finite_map(values, "features")

    @model_validator(mode="after")
    def validate_quality_and_counts(self) -> FeatureRecord:
        if set(self.sample_counts) != set(self.features):
            raise ValueError("sample_counts keys must exactly match feature keys")
        expected = (
            "insufficient"
            if any(count.valid == 0 for count in self.sample_counts.values())
            else "partial"
            if any(count.excluded > 0 for count in self.sample_counts.values())
            else "complete"
        )
        if self.data_quality != expected:
            raise ValueError(f"data_quality must be {expected!r} for the supplied sample counts")
        if any((value is None) != (self.sample_counts[name].valid == 0) for name, value in self.features.items()):
            raise ValueError("feature values must be None exactly when their valid sample count is zero")
        return self


class ProbeVariant(DomainModel):
    variant_id: str = Field(min_length=1)
    base_parameter_set_id: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    parameter_name: ControlParameterName
    direction: Literal["+1", "-1"]
    parameter_set: ParameterSet


class ProbePlan(DomainModel):
    baseline_id: str = Field(min_length=1)
    baseline: ParameterSet
    target_cells: list[str] = Field(min_length=1, max_length=MAX_TARGET_CELLS)
    variants: list[ProbeVariant] = Field(default_factory=list, max_length=MAX_TARGET_CELLS * 10)

    @model_validator(mode="after")
    def validate_plan(self) -> ProbePlan:
        if len(set(self.target_cells)) != len(self.target_cells):
            raise ValueError("target_cells must be unique")
        if set(self.target_cells) != set(self.baseline.cells):
            raise ValueError("target_cells must exactly match baseline cells")
        variant_ids = [variant.variant_id for variant in self.variants]
        if len(set(variant_ids)) != len(variant_ids):
            raise ValueError("variant_id values must be unique")
        for variant in self.variants:
            if variant.base_parameter_set_id != self.baseline_id:
                raise ValueError("every variant must reference baseline_id")
            if variant.cell_id not in self.target_cells:
                raise ValueError("every variant cell_id must be a target cell")
            if set(variant.parameter_set.cells) != set(self.target_cells):
                raise ValueError("every variant must contain exactly the target cells")
        return self


class ThresholdKpi(DomainModel):
    target_kpi: str = Field(min_length=1)
    lower_bound: float | None = None
    upper_bound: float | None = None
    improve_direction: Literal["higher_is_better", "lower_is_better"]

    @model_validator(mode="after")
    def validate_bounds(self) -> ThresholdKpi:
        if self.lower_bound is None and self.upper_bound is None:
            raise ValueError("at least one KPI bound is required")
        for name in ("lower_bound", "upper_bound"):
            value = getattr(self, name)
            if value is not None:
                _finite(value, name)
        if self.lower_bound is not None and self.upper_bound is not None and self.lower_bound > self.upper_bound:
            raise ValueError("lower_bound cannot exceed upper_bound")
        return self


class TimeStepAssignment(DomainModel):
    time_step: int = Field(ge=0, le=4)
    xapp_objective: XAppObjective
    parameter_set: ParameterSet
    predicted_kpis: dict[str, dict[str, float]]
    model_name: str = Field(min_length=1, max_length=128)
    model_version: int = Field(ge=1)
    threshold_violations: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("predicted_kpis")
    @classmethod
    def validate_predictions(cls, values: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        if any(not kpi or not cell_values for kpi, cell_values in values.items()):
            raise ValueError("predicted KPI and cell maps must be non-empty")
        for kpi, cell_values in values.items():
            for cell_id, value in cell_values.items():
                if not cell_id:
                    raise ValueError("predicted KPI cell_id must be non-empty")
                _finite(value, f"predicted_kpis.{kpi}.{cell_id}")
        return values


class TemporalPlan(DomainModel):
    assignments: list[TimeStepAssignment] = Field(min_length=5, max_length=5)
    aggregate_kpis: dict[str, float]
    transition_time_steps: list[int] = Field(default_factory=list)

    @field_validator("aggregate_kpis")
    @classmethod
    def validate_aggregates(cls, values: dict[str, float]) -> dict[str, float]:
        _validate_named_finite_map(values, "aggregate_kpis")
        return values

    @model_validator(mode="after")
    def validate_sequence(self) -> TemporalPlan:
        if [assignment.time_step for assignment in self.assignments] != list(range(5)):
            raise ValueError("assignments must contain time_step 0..4 in ascending order")
        expected_transitions = [
            index
            for index in range(1, 5)
            if self.assignments[index - 1].xapp_objective != self.assignments[index].xapp_objective
        ]
        if self.transition_time_steps != expected_transitions:
            raise ValueError("transition_time_steps must exactly match objective changes")
        return self


class StageRecord(DomainModel):
    stage: Literal["extract_features", "train_model", "save_artifact", "register_metrics"]
    started_at: datetime
    ended_at: datetime | None = None
    result: Literal["succeeded", "failed"] | None = None
    reason: str | None = None

    @field_validator("started_at", "ended_at")
    @classmethod
    def validate_timestamps(cls, value: datetime | None, info: Any) -> datetime | None:
        return None if value is None else _validate_utc(value, info.field_name)

    @model_validator(mode="after")
    def validate_completion(self) -> StageRecord:
        if (self.ended_at is None) != (self.result is None):
            raise ValueError("ended_at and result must be set together")
        if self.ended_at is not None and self.ended_at < self.started_at:
            raise ValueError("ended_at cannot precede started_at")
        if self.reason is not None and self.result != "failed":
            raise ValueError("reason may only be set for a failed stage")
        return self


class TrainingJob(DomainModel):
    job_id: str = Field(min_length=1)
    feature_group: str = Field(min_length=1, max_length=128)
    model_name: str = Field(min_length=1, max_length=128)
    seed: int = Field(ge=0, le=MAX_SEED)
    status: Literal["pending", "running", "completed", "failed"]
    current_stage: str | None = None
    stage_history: list[StageRecord] = Field(default_factory=list)
    failure_type: Literal["data_error", "training_error", "storage_error"] | None = None
    failure_reason: str | None = None
    metrics_registration_failure_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    training_summary: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_status_fields(self) -> TrainingJob:
        if self.status == "running" and self.current_stage is None:
            raise ValueError("running jobs require current_stage")
        if self.status != "running" and self.current_stage is not None:
            raise ValueError("only running jobs may have current_stage")
        if self.status == "failed" and self.failure_type is None:
            raise ValueError("failed jobs require failure_type")
        if self.status != "failed" and (self.failure_type is not None or self.failure_reason is not None):
            raise ValueError("only failed jobs may have failure metadata")
        if self.metrics_registration_failure_reason is not None and self.status != "completed":
            raise ValueError("metrics_registration_failure_reason requires a completed job")
        return self


class ModelVersionRecord(DomainModel):
    model_name: str = Field(min_length=1, max_length=128)
    version: int = Field(ge=1)
    created_at: datetime
    feature_group: str = Field(min_length=1, max_length=128)
    metrics: dict[str, float]
    artifact_uri: str = Field(min_length=1)
    # Requirement 3.4 completion metadata: recorded whenever the Training_Manager
    # registers a completed job's metrics. Optional (default None) so callers that
    # register versions outside the training pipeline (e.g. manual registration)
    # are unaffected.
    train_split: float | None = Field(default=None, ge=0.0, le=1.0)
    validation_split: float | None = Field(default=None, ge=0.0, le=1.0)
    train_samples: int | None = Field(default=None, ge=0)
    validation_samples: int | None = Field(default=None, ge=0)
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return _validate_utc(value, "created_at")

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, values: dict[str, float]) -> dict[str, float]:
        if not values:
            raise ValueError("metrics must be non-empty")
        if not set(values).issubset(TARGET_KPIS):
            raise ValueError("metrics contains an undefined Target_KPI")
        for kpi, value in values.items():
            _finite(value, f"metrics.{kpi}")
            if value < 0:
                raise ValueError("MAPE metrics cannot be negative")
        return values


class MarginalEffectRecord(DomainModel):
    cell_id: str = Field(min_length=1)
    control_parameter: ControlParameterName
    target_kpi: str = Field(min_length=1)
    value_percent: float | None = Field(default=None, ge=-100.0, le=100.0)


class EvidenceRecord(DomainModel):
    evidence_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1, max_length=128)
    model_version: int = Field(ge=1)
    probe_plan: ProbePlan
    predictions: list[dict[str, Any]]
    applied_thresholds: list[ThresholdKpi]
    degradation_verdict: DegradationVerdict
    marginal_effects: list[MarginalEffectRecord] | None = None
    conflict_threshold: float | None = Field(default=None, ge=0.1, le=100.0)
    indirect_conflicts: list[dict[str, Any]] | None = None
    judgment_details: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[dict[str, Any]] = Field(default_factory=list)
    temporal_plan: dict[str, Any] | None = None
    temporal_plan_ref: str | None = None
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return _validate_utc(value, "created_at")

    @model_validator(mode="after")
    def validate_evidence(self) -> EvidenceRecord:
        if not self.predictions:
            raise ValueError("predictions cannot be empty")
        if (self.marginal_effects is None) != (self.conflict_threshold is None):
            raise ValueError("marginal_effects and conflict_threshold must be set together")
        return self
