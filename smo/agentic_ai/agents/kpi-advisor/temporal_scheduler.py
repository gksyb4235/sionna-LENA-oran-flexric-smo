"""Temporal_Scheduler: Time_Step 0..4 xApp execution planning (design.md 10, Requirement 10).

``build_plan`` evaluates every (Time_Step, xApp_Objective, candidate Parameter_Set)
combination through GNN_MCP_Server and assembles a ``TemporalPlan`` with exactly
5 ``TimeStepAssignment`` entries (Time_Step 0..4, ascending -- Requirement 10.10,
already re-enforced by ``TemporalPlan``'s own Pydantic validator in
``smo.aimlfw.common.models``).

Two assignment modes (Requirement 10.1/10.2):

- **Objective-driven** (1..3 ``xApp_Objective`` values given, each with 1..64
  candidate Parameter_Sets): for every Time_Step, the single (objective,
  candidate) combination whose *own* objective's primary Target_KPI value is
  best (max for ``higher_is_better``, min for ``lower_is_better``) is chosen.
  Because different objectives can have different primary KPIs (and
  different improvement directions), comparing across objectives requires a
  single unified "higher is better" scale -- this module negates
  ``lower_is_better`` values before comparing (see ``_unified_criterion``).
- **Unspecified** (no objectives given, only pooled candidate Parameter_Sets):
  every Time_Step is assigned objective ``"unspecified"`` and the candidate
  with the fewest Threshold_KPI violations at that Time_Step.

Both modes reuse ``agent.py``'s in-process GNN_MCP_Server calling convention
(``predict_batch`` defaulting to ``mcp_server.predict_batch``, Requirement
7.9's "only through GNN_MCP_Server" constraint), its ``split_into_batches``
<=64-sized batching helper, and its ``_threshold_violation`` Threshold_KPI
boundary check (Requirement 8's, reused here for 10.2/10.6) -- imported
directly from ``agent`` rather than duplicated, per this module's sibling
package convention (``mcp_server.py``, ``agent.py`` already sit side by side
as plain top-level modules in this directory).

**Unlike Requirement 7.11's Probe_Plan execution, Requirement 10.13 defines
no batch retry policy for Temporal_Scheduler** -- a single GNN_MCP_Server
tool-result error or timeout immediately fails the whole ``build_plan`` call
with ``prediction_unavailable`` and no partial ``TemporalPlan`` is ever
returned, matching design.md's sequence diagram (no retry loop drawn for
Temporal_Scheduler, unlike Probe_Plan's).

**Requirement 10.12 ("time_step_out_of_range") is not reachable from this
function's contract** -- ``build_plan`` takes no caller-supplied Time_Step
argument at all (per design.md's fixed signature); it always evaluates the
full 0..4 range unconditionally, and its return type (``TemporalPlan``) is
structurally validated to always contain exactly Time_Step 0..4. There is no
input value this function could reject as "out of range" without also
breaking that structural invariant. This AC only makes sense at a higher
layer that accepts a caller-restricted Time_Step subset/index (e.g. a future
``KPI_Advisor_Agent`` request field or a thin wrapper around ``build_plan``)
-- that layer does not exist yet in this codebase, so ``TemporalSchedulerError``
still defines a ``"time_step_out_of_range"`` error code for that future
caller to raise, but nothing in this module raises it today.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from smo.aimlfw.common.config import (
    KpiThresholdConfig,
    ObjectivePrimaryKpiConfig,
    TargetKpiAggregationConfig,
)
from smo.aimlfw.common.constants import MAX_TIME_STEP, MIN_TIME_STEP, TARGET_KPIS
from smo.aimlfw.common.errors import ErrorResponse
from smo.aimlfw.common.models import CellParameters, ParameterSet, TemporalPlan, TimeStepAssignment
from smo.aimlfw.feature_store import FeatureStore

import mcp_server
from agent import (
    ProbePredictionResult,
    ThresholdViolation,
    _is_tool_error,
    _threshold_violation,
    _to_prediction_result,
    split_into_batches,
)

# Requirement 10.1: 1..3 xApp_Objective values.
MIN_OBJECTIVES = 1
MAX_OBJECTIVES = 3
# Requirement 10.1/10.3: 1..64 candidate Parameter_Sets per xApp_Objective per batch.
MIN_CANDIDATES_PER_OBJECTIVE = 1
MAX_CANDIDATES_PER_OBJECTIVE = 64


class TemporalSchedulerError(Exception):
    """An expected Temporal_Scheduler failure with a stable ``ErrorResponse``-shaped body.

    Mirrors ``agent.ProbeExecutionError``'s convention exactly: a stable
    ``error_code``/``message``/``details`` body rather than an ad-hoc dict or
    bare exception, so a future caller (e.g. ``agent.py``'s eventual
    Temporal_Plan orchestration, or ``server.py``) can catch one exception
    type and reuse ``details`` unchanged.
    """

    def __init__(self, error_code: str, message: str, details: dict[str, Any] | None = None):
        self.response = ErrorResponse(error_code=error_code, message=message, details=details or {})
        super().__init__(message)

    @property
    def error_code(self) -> str:
        return self.response.error_code

    @property
    def details(self) -> dict[str, Any]:
        return self.response.details


@dataclasses.dataclass(frozen=True)
class _Combo:
    """One (xApp_Objective, candidate Parameter_Set) evaluation point for one Time_Step batch."""

    objective: str
    candidate_index: int  # position within candidates_by_objective[objective] (input order, Requirement 10.11)
    parameter_set: ParameterSet


def _unified_criterion(value: float, improve_direction: str) -> float:
    """Map a Target_KPI value onto a single "higher is better" scale (Requirement 10.1).

    ``higher_is_better`` values are used as-is; ``lower_is_better`` values are
    negated so that, across objectives with differing primary KPIs and
    improvement directions, ``max()`` over this unified criterion always
    picks the combination whose own objective considers its own primary KPI
    value best.
    """
    return value if improve_direction == "higher_is_better" else -value


def _violation_count(prediction: ProbePredictionResult, thresholds: KpiThresholdConfig) -> int:
    """Count how many Target_KPIs of ``prediction`` violate their Threshold_KPI bounds (Requirement 10.2/10.6)."""
    return sum(
        1
        for target_kpi in TARGET_KPIS
        if _threshold_violation(target_kpi, prediction.target_kpi[target_kpi], thresholds.thresholds[target_kpi])
        is not None
    )


def _all_violations(prediction: ProbePredictionResult, thresholds: KpiThresholdConfig) -> list[ThresholdViolation]:
    """Every Threshold_KPI violation in ``prediction`` (Requirement 10.6, unbounded -- unlike 8.3's top-10 cap)."""
    violations = []
    for target_kpi in TARGET_KPIS:
        threshold = thresholds.thresholds[target_kpi]
        violation = _threshold_violation(target_kpi, prediction.target_kpi[target_kpi], threshold)
        if violation is not None:
            violations.append(violation)
    return violations


def _transpose_cell_kpi(cell_kpi: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Reshape Inference_Service's ``{cell_id: {target_kpi: value}}`` into ``TimeStepAssignment``'s
    ``predicted_kpis`` shape, ``{target_kpi: {cell_id: value}}`` (Requirement 10.4).
    """
    predicted_kpis: dict[str, dict[str, float]] = {}
    for cell_id, kpi_values in cell_kpi.items():
        for target_kpi, value in kpi_values.items():
            predicted_kpis.setdefault(target_kpi, {})[cell_id] = value
    return predicted_kpis


def _build_objective_combos(
    objectives: list[str],
    candidates_by_objective: dict[str, list[ParameterSet]],
    objective_primary_kpi: ObjectivePrimaryKpiConfig,
) -> list[_Combo]:
    """Validate and flatten the objective-driven (1..3 objectives) candidate universe (Requirement 10.1)."""
    if not MIN_OBJECTIVES <= len(objectives) <= MAX_OBJECTIVES:
        raise TemporalSchedulerError(
            "invalid_request",
            f"objectives must contain {MIN_OBJECTIVES}..{MAX_OBJECTIVES} entries",
            {"objective_count": len(objectives)},
        )
    unknown = sorted(set(objectives) - set(objective_primary_kpi.objectives))
    if unknown:
        raise TemporalSchedulerError(
            "invalid_request", "objectives contains unrecognized xApp_Objective name(s)", {"unknown": unknown}
        )
    combos: list[_Combo] = []
    for objective in objectives:
        candidates = candidates_by_objective.get(objective, [])
        if not MIN_CANDIDATES_PER_OBJECTIVE <= len(candidates) <= MAX_CANDIDATES_PER_OBJECTIVE:
            raise TemporalSchedulerError(
                "invalid_request",
                f"objective {objective!r} must have {MIN_CANDIDATES_PER_OBJECTIVE}.."
                f"{MAX_CANDIDATES_PER_OBJECTIVE} candidate Parameter_Sets",
                {"objective": objective, "candidate_count": len(candidates)},
            )
        combos.extend(
            _Combo(objective=objective, candidate_index=index, parameter_set=candidate)
            for index, candidate in enumerate(candidates)
        )
    return combos


def _derive_candidates_from_feature_store(
    feature_store: FeatureStore | None, feature_group: str
) -> dict[int, ParameterSet]:
    """Derive one Parameter_Set candidate per Time_Step from Feature_Store (Requirement 10.8, 10.9).

    Only ``data_quality != "insufficient"`` Feature_Records are eligible.
    Target cells are taken as the union of cell_ids across every qualifying
    record (a Feature_Group's target cell set is fixed, so this recovers it
    without needing a separate ``target_cells`` argument). For each Time_Step,
    every target cell must have a qualifying Feature_Record at that exact
    Time_Step to form a complete Parameter_Set (``ParameterSet`` requires
    every cell); Feature_Store's own per-(Time_Step, cell_id) uniqueness
    invariant (Requirement 1.1) means there is normally at most one candidate
    record per cell per Time_Step, so no "best of several" comparison is
    actually needed here in practice.

    Raises ``TemporalSchedulerError("no_candidate", ...)`` if there are 0
    qualifying records at all, or if any Time_Step 0..4 cannot be fully
    covered across every target cell -- a partial Temporal_Plan would violate
    Requirement 10.10's "정확히 5개 항목" invariant, so this is treated as
    equivalent to having no usable candidates rather than as a spec case of
    its own.
    """
    if feature_store is None:
        raise TemporalSchedulerError(
            "no_candidate", "No candidate Parameter_Sets were supplied and no Feature_Store is configured"
        )
    qualifying = [record for record in feature_store.records(feature_group) if record.data_quality != "insufficient"]
    if not qualifying:
        raise TemporalSchedulerError(
            "no_candidate", "Feature_Store has no non-insufficient Feature_Record to derive candidates from"
        )
    target_cells = sorted({record.cell_id for record in qualifying})
    by_step_and_cell: dict[tuple[int, str], CellParameters] = {
        (record.time_step, record.cell_id): record.control_parameters for record in qualifying
    }
    derived: dict[int, ParameterSet] = {}
    for time_step in range(MIN_TIME_STEP, MAX_TIME_STEP + 1):
        cells: dict[str, CellParameters] = {}
        for cell_id in target_cells:
            control_parameters = by_step_and_cell.get((time_step, cell_id))
            if control_parameters is None:
                raise TemporalSchedulerError(
                    "no_candidate",
                    f"Feature_Store has no non-insufficient Feature_Record for cell {cell_id!r}"
                    f" at Time_Step {time_step}",
                    {"time_step": time_step, "cell_id": cell_id},
                )
            cells[cell_id] = control_parameters
        derived[time_step] = ParameterSet(cells=cells)
    return derived


def _pool_candidates(candidates_by_objective: dict[str, list[ParameterSet]]) -> list[ParameterSet]:
    """Flatten every supplied candidate list into one pooled, input-ordered list (unspecified-objective path)."""
    pooled: list[ParameterSet] = []
    for candidates in candidates_by_objective.values():
        pooled.extend(candidates)
    return pooled


def _combos_for_time_step(
    objectives: list[str],
    candidates_by_objective: dict[str, list[ParameterSet]],
    objective_primary_kpi: ObjectivePrimaryKpiConfig,
    feature_store: FeatureStore | None,
    feature_group: str,
) -> dict[int, list[_Combo]]:
    """Build the per-Time_Step candidate universe for both assignment modes (Requirement 10.1, 10.2, 10.8)."""
    if objectives:
        combos = _build_objective_combos(objectives, candidates_by_objective, objective_primary_kpi)
        return {time_step: combos for time_step in range(MIN_TIME_STEP, MAX_TIME_STEP + 1)}

    pooled = _pool_candidates(candidates_by_objective)
    if pooled:
        combos = [
            _Combo(objective="unspecified", candidate_index=index, parameter_set=candidate)
            for index, candidate in enumerate(pooled)
        ]
        return {time_step: combos for time_step in range(MIN_TIME_STEP, MAX_TIME_STEP + 1)}

    derived = _derive_candidates_from_feature_store(feature_store, feature_group)
    return {
        time_step: [_Combo(objective="unspecified", candidate_index=0, parameter_set=derived[time_step])]
        for time_step in range(MIN_TIME_STEP, MAX_TIME_STEP + 1)
    }


def _evaluate_time_step(
    time_step: int,
    combos: list[_Combo],
    predict_batch: Callable[..., dict[str, Any]],
) -> tuple[list[ProbePredictionResult], str, int]:
    """Submit every combo's Parameter_Set for one Time_Step in <=64-sized batches (Requirement 10.3, 10.13).

    Raises ``TemporalSchedulerError("prediction_unavailable", ...)`` immediately on the first
    batch error or timeout -- no retry (unlike Requirement 7.11's Probe_Plan execution) and no
    partial Temporal_Plan is ever assembled from a partially-evaluated Time_Step.
    """
    payloads = [{"cells": combo.parameter_set.model_dump(mode="json")["cells"]} for combo in combos]
    batches = split_into_batches(payloads)
    predictions: list[ProbePredictionResult] = []
    model_name: str | None = None
    model_version: int | None = None
    for batch_index, batch in enumerate(batches):
        result = predict_batch(parameter_sets=batch, time_step=time_step)
        if _is_tool_error(result):
            raise TemporalSchedulerError(
                "prediction_unavailable",
                f"GNN_MCP_Server failed to predict Time_Step {time_step} batch {batch_index}",
                {
                    "time_step": time_step,
                    "batch_index": batch_index,
                    "underlying_error_code": result.get("error_code"),
                    "underlying_message": result.get("message"),
                },
            )
        if model_name is None:
            model_name, model_version = result["model_name"], result["model_version"]
        predictions.extend(_to_prediction_result(prediction) for prediction in result["predictions"])
    return predictions, model_name, model_version  # type: ignore[return-value]  # non-empty combos => always set.


def _select_index(
    combos: list[_Combo],
    predictions: list[ProbePredictionResult],
    *,
    objectives: list[str],
    objective_primary_kpi: ObjectivePrimaryKpiConfig,
    thresholds: KpiThresholdConfig,
) -> int:
    """Pick the winning combo index for one Time_Step (Requirement 10.1/10.2, tie-broken per 10.11).

    Requirement 10.11's deterministic tie-break (objective name ascending,
    then input order for equal-named ties) is implemented as the secondary
    and tertiary sort keys below; Python's ``sorted`` is stable and total
    ordering on ``(criterion, objective, candidate_index)`` makes the result
    independent of the combos list's original iteration order.
    """
    if objectives:
        def criterion(index: int) -> float:
            combo = combos[index]
            entry = objective_primary_kpi.objectives[combo.objective]
            value = predictions[index].target_kpi[entry.target_kpi]
            return -_unified_criterion(value, entry.improve_direction)  # negate: ascending sort => best first.
    else:
        def criterion(index: int) -> float:
            return float(_violation_count(predictions[index], thresholds))

    ordered = sorted(
        range(len(combos)), key=lambda index: (criterion(index), combos[index].objective, combos[index].candidate_index)
    )
    return ordered[0]


def build_plan(
    objectives: list[str],
    candidates_by_objective: dict[str, list[ParameterSet]],
    *,
    thresholds: KpiThresholdConfig,
    objective_primary_kpi: ObjectivePrimaryKpiConfig,
    target_kpi_aggregation: TargetKpiAggregationConfig,
    feature_store: FeatureStore | None = None,
    feature_group: str = "default",
    predict_batch: Callable[..., dict[str, Any]] = mcp_server.predict_batch,
) -> TemporalPlan:
    """Build a 5-entry Temporal_Plan across Time_Step 0..4 (Requirement 10.1-10.13).

    - ``objectives`` empty => "unspecified"-objective mode (Requirement 10.2): every
      Time_Step is assigned the pooled candidate with the fewest Threshold_KPI
      violations, or (if ``candidates_by_objective`` is also empty/all-empty) a
      Feature_Store-derived candidate (Requirement 10.8/10.9).
    - ``objectives`` non-empty (1..3 entries) => objective-driven mode
      (Requirement 10.1): every Time_Step is assigned the single (objective,
      candidate) combination whose own objective's primary Target_KPI value
      is best.

    Raises ``TemporalSchedulerError`` with error_code:
      - ``"invalid_request"``: objective count/candidate count outside the
        1..3 / 1..64 ranges Requirement 10.1 defines, or an unrecognized
        objective name -- defensive input validation outside Requirement
        10's own error catalog, mirroring ``agent.py``'s ``empty_request``-
        style validation for cases the Requirement text does not name.
      - ``"no_candidate"`` (Requirement 10.9): zero candidates given and
        Feature_Store has no usable fallback data.
      - ``"prediction_unavailable"`` (Requirement 10.13): GNN_MCP_Server
        returned an error or timed out for any batch; no partial plan.
    """
    group = feature_group
    combos_by_step = _combos_for_time_step(
        objectives, candidates_by_objective, objective_primary_kpi, feature_store, group
    )

    assignments: list[TimeStepAssignment] = []
    raw_target_kpi_by_step: list[dict[str, float]] = []
    for time_step in range(MIN_TIME_STEP, MAX_TIME_STEP + 1):
        combos = combos_by_step[time_step]
        predictions, model_name, model_version = _evaluate_time_step(time_step, combos, predict_batch)
        chosen_index = _select_index(
            combos,
            predictions,
            objectives=objectives,
            objective_primary_kpi=objective_primary_kpi,
            thresholds=thresholds,
        )
        chosen_combo = combos[chosen_index]
        chosen_prediction = predictions[chosen_index]
        violations = _all_violations(chosen_prediction, thresholds)

        assignments.append(
            TimeStepAssignment(
                time_step=time_step,
                xapp_objective=chosen_combo.objective,
                parameter_set=chosen_combo.parameter_set,
                predicted_kpis=_transpose_cell_kpi(chosen_prediction.cell_kpi),
                model_name=model_name,
                model_version=model_version,
                threshold_violations=[dataclasses.asdict(violation) for violation in violations],
            )
        )
        raw_target_kpi_by_step.append(chosen_prediction.target_kpi)

    aggregate_kpis: dict[str, float] = {}
    for target_kpi in TARGET_KPIS:
        values = [raw_target_kpi_by_step[time_step][target_kpi] for time_step in range(5)]
        rule = target_kpi_aggregation.aggregations[target_kpi]
        aggregate_kpis[target_kpi] = sum(values) if rule == "sum" else sum(values) / len(values)

    transition_time_steps = [
        index for index in range(1, 5) if assignments[index - 1].xapp_objective != assignments[index].xapp_objective
    ]

    return TemporalPlan(
        assignments=assignments, aggregate_kpis=aggregate_kpis, transition_time_steps=transition_time_steps
    )


__all__ = ["TemporalSchedulerError", "build_plan"]
