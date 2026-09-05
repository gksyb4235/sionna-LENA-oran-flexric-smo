"""Canonical Hypothesis strategies shared by SMO property tests.

Property modules use ``PROPERTY_TEST_SETTINGS`` and retain the design tags::

    # **Property N: property name**
    # **Validates: Requirements X.Y**
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy

from smo.aimlfw.common.constants import (
    CONTROL_PARAMETER_NAMES,
    CONTROL_PARAMETER_RANGES,
    MAX_SEED,
    MAX_TARGET_CELLS,
    TARGET_KPIS,
    TTT_ALLOWED_MS,
)
from smo.aimlfw.common.models import (
    CellParameters,
    FeatureRecord,
    ParameterSet,
    ProbePlan,
    ProbeVariant,
    SampleCount,
    TemporalPlan,
    ThresholdKpi,
    TimeStepAssignment,
)

MAX_EXAMPLES = 20
PROPERTY_TEST_SETTINGS = settings(max_examples=MAX_EXAMPLES, deadline=None)

_CELL_ID = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
    min_size=1,
    max_size=32,
).filter(str.strip)
_FINITE_FLOAT = st.floats(
    min_value=-1_000_000.0,
    max_value=1_000_000.0,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)

def cell_ids(*, min_size: int = 1, max_size: int = MAX_TARGET_CELLS) -> SearchStrategy[list[str]]:
    """Generate unique external cell identifiers."""
    if not 1 <= min_size <= max_size <= MAX_TARGET_CELLS:
        raise ValueError("cell count bounds must satisfy 1 <= min_size <= max_size <= 16")
    return st.lists(_CELL_ID, min_size=min_size, max_size=max_size, unique=True)


@st.composite
def valid_cell_parameters(draw: Any) -> CellParameters:
    """Generate one complete, range- and step-aligned cell parameter set."""
    return CellParameters(
        tx_power_dbm=float(draw(st.integers(min_value=30, max_value=46))),
        ret_tilt_deg=float(draw(st.integers(min_value=0, max_value=15))),
        cio_bias_db=draw(st.integers(min_value=-12, max_value=12)) / 2.0,
        hysteresis_db=draw(st.integers(min_value=0, max_value=20)) / 2.0,
        ttt_ms=draw(st.sampled_from(TTT_ALLOWED_MS)),
    )


_INVALID_PARAMETER_VALUES: dict[str, SearchStrategy[float | int]] = {
    "tx_power_dbm": st.sampled_from((29.0, 30.5, 47.0)),
    "ret_tilt_deg": st.sampled_from((-1.0, 1.5, 16.0)),
    "cio_bias_db": st.sampled_from((-6.5, 0.25, 6.5)),
    "hysteresis_db": st.sampled_from((-0.5, 0.25, 10.5)),
    "ttt_ms": st.sampled_from((-1, 1, 200, 5121)),
}


@st.composite
def invalid_cell_parameters(draw: Any) -> dict[str, float | int]:
    """Generate a complete cell payload with exactly one invalid control value."""
    values = draw(valid_cell_parameters()).model_dump()
    parameter_name = draw(st.sampled_from(CONTROL_PARAMETER_NAMES))
    values[parameter_name] = draw(_INVALID_PARAMETER_VALUES[parameter_name])
    return values


@st.composite
def valid_parameter_sets(
    draw: Any,
    *,
    min_cells: int = 1,
    max_cells: int = MAX_TARGET_CELLS,
) -> ParameterSet:
    """Generate a complete cell-specific ``ParameterSet``."""
    identifiers = draw(cell_ids(min_size=min_cells, max_size=max_cells))
    parameters = draw(
        st.lists(valid_cell_parameters(), min_size=len(identifiers), max_size=len(identifiers))
    )
    return ParameterSet(cells=dict(zip(identifiers, parameters, strict=True)))


@st.composite
def invalid_parameter_sets(draw: Any) -> dict[str, Any]:
    """Generate payloads rejected by ``ParameterSet`` without constructing invalid models."""
    case = draw(st.sampled_from(("empty", "too_many", "blank_cell", "incomplete", "invalid_value")))
    if case == "empty":
        return {"cells": {}}
    if case == "too_many":
        parameter = draw(valid_cell_parameters()).model_dump()
        return {"cells": {f"cell_{index}": parameter for index in range(MAX_TARGET_CELLS + 1)}}

    parameter = draw(valid_cell_parameters()).model_dump()
    if case == "blank_cell":
        return {"cells": {" ": parameter}}
    if case == "incomplete":
        parameter.pop(draw(st.sampled_from(CONTROL_PARAMETER_NAMES)))
        return {"cells": {"gNB_5G": parameter}}
    return {"cells": {"gNB_5G": draw(invalid_cell_parameters())}}


@st.composite
def feature_records(
    draw: Any,
    *,
    feature_names: Sequence[str] = TARGET_KPIS,
) -> FeatureRecord:
    """Generate internally consistent feature values, counts, and data quality."""
    names = tuple(feature_names)
    if not names or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("feature_names must contain unique, non-empty names")

    counts: dict[str, SampleCount] = {}
    values: dict[str, float | None] = {}
    for name in names:
        valid = draw(st.integers(min_value=0, max_value=1_000))
        excluded = draw(st.integers(min_value=0, max_value=1_000))
        counts[name] = SampleCount(valid=valid, excluded=excluded)
        values[name] = None if valid == 0 else round(draw(_FINITE_FLOAT), 6)

    quality = (
        "insufficient"
        if any(count.valid == 0 for count in counts.values())
        else "partial"
        if any(count.excluded > 0 for count in counts.values())
        else "complete"
    )
    return FeatureRecord(
        feature_group=draw(st.sampled_from(("default", "mobility", "energy"))),
        time_step=draw(st.integers(min_value=0, max_value=4)),
        cell_id=draw(_CELL_ID),
        control_parameters=draw(valid_cell_parameters()),
        features=values,
        sample_counts=counts,
        data_quality=quality,
        source_dir=f"/tmp/results/{draw(_CELL_ID)}",
        seed=draw(st.one_of(st.none(), st.integers(min_value=0, max_value=MAX_SEED))),
    )


def valid_csv_values() -> SearchStrategy[str]:
    """Generate finite values as they can appear in a CSV field."""
    return _FINITE_FLOAT.map(repr)


def invalid_csv_values() -> SearchStrategy[str]:
    """Generate values excluded from numeric CSV aggregation."""
    return st.sampled_from(("", " ", "nan", "NaN", "not-a-number", "Infinity", "-Infinity"))


def csv_values() -> SearchStrategy[str]:
    """Generate arbitrary valid or excluded CSV field values."""
    return st.one_of(valid_csv_values(), invalid_csv_values())


@st.composite
def threshold_kpis(draw: Any, target_kpi: str | None = None) -> ThresholdKpi:
    """Generate a valid one-sided or two-sided KPI threshold."""
    kpi = target_kpi or draw(st.sampled_from(TARGET_KPIS))
    if kpi not in TARGET_KPIS:
        raise ValueError(f"unknown Target_KPI: {kpi}")
    bound_type = draw(st.sampled_from(("lower", "upper", "both")))
    lower = draw(st.floats(-10_000, 10_000, allow_nan=False, allow_infinity=False))
    upper = draw(st.floats(lower, 10_000, allow_nan=False, allow_infinity=False))
    return ThresholdKpi(
        target_kpi=kpi,
        lower_bound=lower if bound_type != "upper" else None,
        upper_bound=upper if bound_type != "lower" else None,
        improve_direction=draw(st.sampled_from(("higher_is_better", "lower_is_better"))),
    )


@st.composite
def threshold_maps(draw: Any) -> dict[str, ThresholdKpi]:
    """Generate a complete canonical Target_KPI threshold mapping."""
    return {kpi: draw(threshold_kpis(kpi)) for kpi in TARGET_KPIS}


def _adjacent_parameter(
    parameters: CellParameters,
    parameter_name: str,
    direction: str,
) -> CellParameters | None:
    values = parameters.model_dump()
    current = values[parameter_name]
    if parameter_name == "ttt_ms":
        current_index = TTT_ALLOWED_MS.index(current)
        next_index = current_index + (1 if direction == "+1" else -1)
        if not 0 <= next_index < len(TTT_ALLOWED_MS):
            return None
        values[parameter_name] = TTT_ALLOWED_MS[next_index]
    else:
        rule = CONTROL_PARAMETER_RANGES[parameter_name]
        delta = float(rule["step"]) * (1 if direction == "+1" else -1)
        candidate = float(current) + delta
        if not float(rule["min"]) <= candidate <= float(rule["max"]):
            return None
        values[parameter_name] = candidate
    return CellParameters.model_validate(values)


@st.composite
def probe_plans(draw: Any) -> ProbePlan:
    """Generate plans whose variants change exactly one cell parameter by one step."""
    baseline = draw(valid_parameter_sets())
    baseline_id = draw(
        st.text(
            alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_",
            min_size=1,
            max_size=24,
        )
    )
    candidates: list[tuple[str, str, str, CellParameters]] = []
    for cell_id, parameters in baseline.cells.items():
        for parameter_name in CONTROL_PARAMETER_NAMES:
            for direction in ("+1", "-1"):
                adjacent = _adjacent_parameter(parameters, parameter_name, direction)
                if adjacent is not None:
                    candidates.append((cell_id, parameter_name, direction, adjacent))

    selected = draw(
        st.lists(
            st.integers(min_value=0, max_value=len(candidates) - 1),
            max_size=len(candidates),
            unique=True,
        )
    )
    variants: list[ProbeVariant] = []
    for candidate_index in selected:
        cell_id, parameter_name, direction, adjacent = candidates[candidate_index]
        cells = dict(baseline.cells)
        cells[cell_id] = adjacent
        variants.append(
            ProbeVariant(
                variant_id=f"variant-{candidate_index}",
                base_parameter_set_id=baseline_id,
                cell_id=cell_id,
                parameter_name=parameter_name,
                direction=direction,
                parameter_set=ParameterSet(cells=cells),
            )
        )
    return ProbePlan(
        baseline_id=baseline_id,
        baseline=baseline,
        target_cells=list(baseline.cells),
        variants=variants,
    )


@st.composite
def temporal_plans(draw: Any) -> TemporalPlan:
    """Generate structurally complete five-step temporal plans."""
    parameter_set = draw(valid_parameter_sets())
    objectives = draw(
        st.lists(
            st.sampled_from(
                ("energy_saving", "throughput_maximization", "mobility_robustness", "unspecified")
            ),
            min_size=5,
            max_size=5,
        )
    )
    model_name = draw(st.sampled_from(("gnn", "ran-gnn", "candidate-model")))
    model_version = draw(st.integers(min_value=1, max_value=10_000))
    assignments: list[TimeStepAssignment] = []
    for time_step, objective in enumerate(objectives):
        predictions = {
            kpi: {cell_id: draw(_FINITE_FLOAT) for cell_id in parameter_set.cells}
            for kpi in TARGET_KPIS
        }
        assignments.append(
            TimeStepAssignment(
                time_step=time_step,
                xapp_objective=objective,
                parameter_set=parameter_set,
                predicted_kpis=predictions,
                model_name=model_name,
                model_version=model_version,
                threshold_violations=[],
            )
        )
    transitions = [
        index for index in range(1, 5) if objectives[index - 1] != objectives[index]
    ]
    return TemporalPlan(
        assignments=assignments,
        aggregate_kpis={kpi: draw(_FINITE_FLOAT) for kpi in TARGET_KPIS},
        transition_time_steps=transitions,
    )


__all__ = [
    "MAX_EXAMPLES",
    "PROPERTY_TEST_SETTINGS",
    "cell_ids",
    "csv_values",
    "feature_records",
    "invalid_cell_parameters",
    "invalid_csv_values",
    "invalid_parameter_sets",
    "probe_plans",
    "temporal_plans",
    "threshold_kpis",
    "threshold_maps",
    "valid_cell_parameters",
    "valid_csv_values",
    "valid_parameter_sets",
]
