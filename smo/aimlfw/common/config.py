"""Validated configuration models and JSON/YAML loaders."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError, model_validator

from .constants import (
    DEFAULT_CONFLICT_THRESHOLD,
    DEFAULT_RETRAINING_THRESHOLD,
    MAX_TARGET_CELLS,
    TARGET_KPIS,
)
from .models import DomainModel, ThresholdKpi

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class ConfigurationError(ValueError):
    """Raised when a configuration file cannot be safely loaded or validated."""

    def __init__(self, path: Path, message: str):
        self.path = path
        super().__init__(f"{path}: {message}")


class FeatureGroupConfig(DomainModel):
    name: str = Field(min_length=1, max_length=128)
    features: list[str] = Field(min_length=1)
    aggregations: dict[str, Literal["sum", "mean"]]
    interval_seconds: int = Field(default=180, ge=1)
    target_cells: list[str] = Field(min_length=1, max_length=MAX_TARGET_CELLS)
    target_kpis: list[str]

    @model_validator(mode="after")
    def validate_contract(self) -> FeatureGroupConfig:
        if len(set(self.features)) != len(self.features):
            raise ValueError("features must be unique")
        if set(self.aggregations) != set(self.features):
            raise ValueError("aggregations must exactly match features")
        if len(set(self.target_cells)) != len(self.target_cells):
            raise ValueError("target_cells must be unique")
        if any(not value.strip() for value in (*self.features, *self.target_cells)):
            raise ValueError("feature and cell names must be non-empty")
        if tuple(self.target_kpis) != TARGET_KPIS:
            raise ValueError("target_kpis must equal the canonical Target_KPI list")
        return self


class KpiThresholdConfig(DomainModel):
    thresholds: dict[str, ThresholdKpi]

    @model_validator(mode="after")
    def validate_all_kpis(self) -> KpiThresholdConfig:
        if set(self.thresholds) != set(TARGET_KPIS):
            missing = sorted(set(TARGET_KPIS) - set(self.thresholds))
            undefined = sorted(set(self.thresholds) - set(TARGET_KPIS))
            raise ValueError(f"threshold KPI mismatch; missing={missing}, undefined={undefined}")
        for name, threshold in self.thresholds.items():
            if threshold.target_kpi != name:
                raise ValueError(f"threshold target_kpi must match key {name!r}")
        return self


class TargetKpiAggregationConfig(DomainModel):
    aggregations: dict[str, Literal["sum", "mean"]]

    @model_validator(mode="after")
    def validate_all_kpis(self) -> TargetKpiAggregationConfig:
        if set(self.aggregations) != set(TARGET_KPIS):
            raise ValueError("KPI aggregations must exactly match the canonical Target_KPI list")
        return self


# Requirement 8.5's "설정 파일에 지정된 주 Target_KPI" only makes sense for an objective-driven
# request; ``unspecified`` (smo.aimlfw.common.constants.XAppObjective) is deliberately excluded
# from this set -- there is no primary KPI to define for the "no objective given" case.
_OBJECTIVE_DRIVEN_XAPP_OBJECTIVES: tuple[str, ...] = (
    "energy_saving",
    "throughput_maximization",
    "mobility_robustness",
)


class ObjectivePrimaryKpiEntry(DomainModel):
    """One xApp_Objective's primary Target_KPI and improvement direction (Requirement 8.5).

    ``improve_direction`` is kept alongside ``target_kpi`` here (rather than always looking it
    up from ``KpiThresholdConfig``) because a KPI's Threshold_KPI improve_direction and its role
    as an objective's primary KPI are conceptually independent settings that happen to coincide
    in this codebase's default configuration (see ``objective_primary_kpi.json``): a future
    configuration could reasonably use a different primary KPI per objective, or even reuse a
    KPI with a differing improvement sense for a specific objective, without touching
    ``kpi_thresholds.json``.
    """

    target_kpi: str = Field(min_length=1)
    improve_direction: Literal["higher_is_better", "lower_is_better"]


class ObjectivePrimaryKpiConfig(DomainModel):
    objectives: dict[str, ObjectivePrimaryKpiEntry]

    @model_validator(mode="after")
    def validate_all_objectives(self) -> ObjectivePrimaryKpiConfig:
        if set(self.objectives) != set(_OBJECTIVE_DRIVEN_XAPP_OBJECTIVES):
            missing = sorted(set(_OBJECTIVE_DRIVEN_XAPP_OBJECTIVES) - set(self.objectives))
            undefined = sorted(set(self.objectives) - set(_OBJECTIVE_DRIVEN_XAPP_OBJECTIVES))
            raise ValueError(f"objective primary KPI mismatch; missing={missing}, undefined={undefined}")
        for objective, entry in self.objectives.items():
            if entry.target_kpi not in TARGET_KPIS:
                raise ValueError(f"{objective} primary target_kpi must be a canonical Target_KPI")
        return self


class RuntimeThresholdConfig(DomainModel):
    conflict_threshold: float = DEFAULT_CONFLICT_THRESHOLD
    retraining_threshold: float = DEFAULT_RETRAINING_THRESHOLD
    warnings: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def apply_defaults(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        values, warnings = dict(raw), list(raw.get("warnings", ()))
        for name, default in (
            ("conflict_threshold", DEFAULT_CONFLICT_THRESHOLD),
            ("retraining_threshold", DEFAULT_RETRAINING_THRESHOLD),
        ):
            value = values.get(name)
            valid = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and 0.1 <= value <= 100.0
            )
            if not valid:
                if name in values or name == "retraining_threshold":
                    warnings.append(f"{name} defaulted to {default}: expected a number in [0.1, 100.0]")
                values[name] = default
        values["warnings"] = tuple(warnings)
        return values


class FrameworkConfig(DomainModel):
    feature_group: FeatureGroupConfig
    kpi_thresholds: KpiThresholdConfig
    target_kpi_aggregation: TargetKpiAggregationConfig
    runtime_thresholds: RuntimeThresholdConfig


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ConfigurationError(path, f"cannot read configuration: {exc}") from exc
    try:
        if path.suffix.lower() == ".json":
            value = json.loads(text)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ConfigurationError(path, "PyYAML is required to load YAML configuration") from exc
            value = yaml.safe_load(text)
        else:
            raise ConfigurationError(path, "supported extensions are .json, .yaml, and .yml")
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(path, f"invalid configuration syntax: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(path, "configuration root must be an object")
    return value


def _validate(model_type: type[DomainModel], data: dict[str, Any], path: Path) -> Any:
    try:
        return model_type.model_validate(data)
    except ValidationError as exc:
        raise ConfigurationError(path, f"configuration validation failed: {exc}") from exc


def load_feature_group(name: str = "default", config_dir: Path | str | None = None) -> FeatureGroupConfig:
    if not _SAFE_NAME.fullmatch(name):
        raise ConfigurationError(Path(name), "invalid feature group name")
    root = Path(config_dir) if config_dir is not None else DEFAULT_CONFIG_DIR / "feature_groups"
    candidates = [root / f"{name}{suffix}" for suffix in (".json", ".yaml", ".yml")]
    path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    return _validate(FeatureGroupConfig, _read_mapping(path), path)


def load_kpi_thresholds(path: Path | str | None = None) -> KpiThresholdConfig:
    source = Path(path) if path is not None else DEFAULT_CONFIG_DIR / "kpi_thresholds.json"
    raw = _read_mapping(source)
    expanded = {
        name: {"target_kpi": name, **value} if isinstance(value, dict) else value
        for name, value in raw.items()
    }
    return _validate(KpiThresholdConfig, {"thresholds": expanded}, source)


def load_target_kpi_aggregation(path: Path | str | None = None) -> TargetKpiAggregationConfig:
    source = Path(path) if path is not None else DEFAULT_CONFIG_DIR / "target_kpi_aggregation.json"
    return _validate(TargetKpiAggregationConfig, {"aggregations": _read_mapping(source)}, source)


def load_objective_primary_kpi(path: Path | str | None = None) -> ObjectivePrimaryKpiConfig:
    """Load the xApp_Objective -> primary Target_KPI mapping (Requirement 8.5)."""
    source = Path(path) if path is not None else DEFAULT_CONFIG_DIR / "objective_primary_kpi.json"
    return _validate(ObjectivePrimaryKpiConfig, {"objectives": _read_mapping(source)}, source)


def load_runtime_thresholds(path: Path | str | None = None) -> RuntimeThresholdConfig:
    source = Path(path) if path is not None else DEFAULT_CONFIG_DIR / "runtime_thresholds.json"
    return _validate(RuntimeThresholdConfig, _read_mapping(source), source)


def load_framework_config(feature_group: str = "default", config_dir: Path | str | None = None) -> FrameworkConfig:
    root = Path(config_dir) if config_dir is not None else DEFAULT_CONFIG_DIR
    return FrameworkConfig(
        feature_group=load_feature_group(feature_group, root / "feature_groups"),
        kpi_thresholds=load_kpi_thresholds(root / "kpi_thresholds.json"),
        target_kpi_aggregation=load_target_kpi_aggregation(root / "target_kpi_aggregation.json"),
        runtime_thresholds=load_runtime_thresholds(root / "runtime_thresholds.json"),
    )
