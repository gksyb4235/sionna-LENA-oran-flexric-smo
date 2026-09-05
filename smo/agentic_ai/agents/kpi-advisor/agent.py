"""KPI Advisor runtime seam.

Task 7.1 defines the server contract and readiness behavior. Task 7.2 adds
Baseline_Parameter_Set construction, Probe_Plan generation, and batched
GNN_MCP_Server execution (Requirement 7). Task 7.3 (this module's remaining
half) adds Requirement 8's Threshold_KPI judgment (acceptable/degrading/
unknown), recommendation-candidate filtering and deterministic ranking, and
wires all of it into an atomically persisted Evidence_Record and the public
``run_advisor`` response contract:

    xapp_request -> build_baseline -> build_probe_plan -> execute_probe_plan
                                                                |
    baseline_parameter_set --------> build_probe_plan --------/
                                                                v
                                            judge_and_recommend -> InvokeResponse

Requirement 7.9 / 12.6-12.8 standing constraint: nothing in this module calls
Policy_Manager or any A1 mutation path. Recommendations here are advisory
data only; actual RAN parameter changes remain entirely out of scope for this
agent's ``/invoke`` response.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import ValidationError
from smo.aimlfw.common.config import (
    ConfigurationError,
    KpiThresholdConfig,
    ObjectivePrimaryKpiConfig,
    ObjectivePrimaryKpiEntry,
    load_kpi_thresholds,
    load_objective_primary_kpi,
    load_runtime_thresholds,
)
from smo.aimlfw.common.constants import CONTROL_PARAMETER_NAMES, TARGET_KPIS, XAppObjective
from smo.aimlfw.common.errors import ErrorResponse
from smo.aimlfw.common.models import (
    CellParameters,
    MarginalEffectRecord,
    ParameterSet,
    ProbePlan,
    ProbeVariant,
    ThresholdKpi,
)
from smo.aimlfw.feature_store import FeatureStore
from smo.aimlfw.inference_service.schemas import MAX_BATCH_SIZE

import mcp_server
from conflict_analyzer import analyze
from evidence_store import assemble_evidence_record, persist_evidence
from probing import STEP_DIRECTIONS, step_value
from schemas import (
    HealthResponse,
    InvokeRequest,
    InvokeResponse,
    Recommendation,
    XAppParameterRequest,
    intent_is_korean,
)

# Requirement 7.11: a failed batch is retried up to 2 additional times (3
# total attempts). This is a *separate*, higher-level retry from
# GNN_MCP_Server's own internal 30s no-retry Inference_Service timeout
# (task 6.1) -- each of these 3 attempts is itself one such 30s-bounded call.
DEFAULT_MAX_BATCH_RETRIES = 2

# smo/aimlfw/feature_store/data, mirroring data_extractor/server.py's and
# training_manager/server.py's DEFAULT_FEATURE_STORE_ROOT convention
# (Path(__file__).resolve().parents[1] / "feature_store" / "data" relative
# to the aimlfw package root; agent.py sits 3 directories below aimlfw's
# parent, hence parents[3]).
DEFAULT_FEATURE_STORE_ROOT = Path(__file__).resolve().parents[3] / "aimlfw" / "feature_store" / "data"
DEFAULT_FEATURE_GROUP = "default"

PercentChange = float | Literal["undefined"]


class AdvisorNotReadyError(RuntimeError):
    """Raised when advisory dependencies are not ready for invocation."""


class ProbeExecutionError(Exception):
    """An expected Baseline/Probe_Plan/batch-execution failure with a stable API error body.

    Mirrors ``FeatureStoreError``/``InferenceServiceError``/``TrainingManagerError``'s
    convention elsewhere in this codebase: a stable ``ErrorResponse``-shaped
    ``error_code``/``message``/``details`` body, rather than an ad-hoc dict. This
    is what lets a future ``run_advisor`` (task 7.3) catch one exception type here
    and reuse the same ``details`` payload it hands back through ``server.py``'s
    existing generic ``except Exception`` -> ``advisor_error`` path (or, once 7.3
    wires it up, a more specific handler keyed off ``error_code``).
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


def get_model_health() -> HealthResponse:
    """Return model readiness from an override or the live Inference Service."""

    model_name = os.getenv("KPI_ADVISOR_MODEL_NAME", "").strip()
    raw_version = os.getenv("KPI_ADVISOR_MODEL_VERSION", "").strip()
    try:
        model_version = int(raw_version)
    except (TypeError, ValueError):
        model_version = 0
    if model_name and model_version >= 1:
        return HealthResponse(status="ready", model_name=model_name, model_version=model_version)
    inference_url = os.getenv("INFERENCE_SERVICE_BASE_URL", "http://127.0.0.1:8105").rstrip("/")
    try:
        response = httpx.get(f"{inference_url}/status", timeout=2.5)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return HealthResponse(status="not_ready", model_name=None, model_version=None)
    if (
        payload.get("model_load_status") == "loaded"
        and isinstance(payload.get("model_name"), str)
        and isinstance(payload.get("model_version"), int)
        and payload["model_version"] >= 1
    ):
        return HealthResponse(
            status="ready",
            model_name=payload["model_name"],
            model_version=payload["model_version"],
        )
    return HealthResponse(status="not_ready", model_name=None, model_version=None)


# ---------------------------------------------------------------------------
# Feature_Store injection (mirrors mcp_server.py's set_client_factory/
# reset_client_factory HTTP-client injection seam, task 6.1) -- lets
# ``build_and_execute_probe_plan`` (and task 7.16's later integration tests)
# substitute a test Feature_Store without threading one through every call.
# ``build_baseline`` itself always takes an explicit ``feature_store``
# argument rather than reaching for this factory directly, so unit tests can
# just pass a fake store in without touching module globals at all.
# ---------------------------------------------------------------------------


def _default_feature_store_factory() -> FeatureStore:
    return FeatureStore(os.getenv("FEATURE_STORE_ROOT", str(DEFAULT_FEATURE_STORE_ROOT)))


_feature_store_factory: Callable[[], FeatureStore] = _default_feature_store_factory


def set_feature_store_factory(factory: Callable[[], FeatureStore]) -> None:
    """Override the default Feature_Store factory (test seam)."""
    global _feature_store_factory
    _feature_store_factory = factory


def reset_feature_store_factory() -> None:
    """Restore the default (env-var configured) Feature_Store factory."""
    global _feature_store_factory
    _feature_store_factory = _default_feature_store_factory


def get_feature_store() -> FeatureStore:
    """Return a Feature_Store instance from the currently configured factory."""
    return _feature_store_factory()


def _feature_group() -> str:
    return os.getenv("KPI_ADVISOR_FEATURE_GROUP", DEFAULT_FEATURE_GROUP)


# ---------------------------------------------------------------------------
# Requirement 7.1: Baseline_Parameter_Set construction.
# ---------------------------------------------------------------------------


def build_baseline(
    xapp_request: XAppParameterRequest,
    feature_store: FeatureStore,
    *,
    feature_group: str | None = None,
) -> ParameterSet:
    """Complete a Baseline_Parameter_Set from xApp overrides plus recent Feature_Records.

    For every target cell, any Control_Parameter the xApp request did not
    override is filled from that cell's most recent Feature_Record -- "가장
    최근" is proxied by the highest ``time_step`` present for the cell, since
    ``FeatureRecord`` (smo.aimlfw.common.models) carries no timestamp finer
    than ``time_step`` (0..4). Every one of the 1..16 ``target_cells`` ends up
    with all 5 Control_Parameters set (Requirement 7.1).

    Raises ``ProbeExecutionError`` (``feature_record_unavailable``) if a cell
    needs a Feature_Record value that Feature_Store cannot supply -- an edge
    case outside Requirement 7's own error catalog (which only defines
    ``empty_request``/``no_prediction_evidence`` for KPI_Advisor_Agent), but
    one this function cannot silently paper over without inventing values.
    """
    group = feature_group if feature_group is not None else _feature_group()
    cells: dict[str, CellParameters] = {}
    for cell_id in xapp_request.target_cells:
        overrides = xapp_request.parameter_overrides.get(cell_id)
        override_values = overrides.model_dump() if overrides is not None else {}
        resolved = dict(override_values)
        missing = [name for name in CONTROL_PARAMETER_NAMES if resolved.get(name) is None]
        if missing:
            records = feature_store.records(group, cell_id=cell_id)
            if not records:
                raise ProbeExecutionError(
                    "feature_record_unavailable",
                    f"No Feature_Record is available to complete cell {cell_id!r}",
                    {"cell_id": cell_id, "missing_parameters": missing},
                )
            latest = max(records, key=lambda record: record.time_step)
            latest_values = latest.control_parameters.model_dump()
            for name in missing:
                resolved[name] = latest_values.get(name)
        still_missing = [name for name in CONTROL_PARAMETER_NAMES if resolved.get(name) is None]
        if still_missing:
            raise ProbeExecutionError(
                "feature_record_unavailable",
                f"Feature_Record for cell {cell_id!r} does not define {still_missing}",
                {"cell_id": cell_id, "missing_parameters": still_missing},
            )
        cells[cell_id] = CellParameters.model_validate(resolved)
    return ParameterSet(cells=cells)


# ---------------------------------------------------------------------------
# Requirement 7.2, 7.3, 7.4: Probe_Plan generation.
# ---------------------------------------------------------------------------


def build_probe_plan(baseline: ParameterSet, baseline_id: str) -> ProbePlan:
    """Build a Probe_Plan of +-1-step Probe_Variants for every (cell, Control_Parameter).

    For every target cell and every Control_Parameter, this generates exactly
    one ``+1`` and one ``-1`` direction Probe_Variant differing from
    ``baseline`` in exactly that one (cell_id, Control_Parameter) value
    (Requirement 7.3), using ``probing.step_value`` for the per-parameter step
    size (TxP 1, RET 1, CIO 0.5, HYS 0.5, TTT adjacent allowed value --
    Requirement 7.2). A direction is skipped whenever the stepped value would
    leave the Control_Parameter's allowed range/step set; if every direction
    for every (cell, parameter) combination is out of range, the returned
    Probe_Plan simply has an empty ``variants`` list (Requirement 7.4) -- this
    is not treated as an error.

    ``ProbePlan``'s own Pydantic validator (smo.aimlfw.common.models) already
    enforces unique ``variant_id`` values, that every variant references
    ``baseline_id``, and that every variant's ``parameter_set`` covers exactly
    the target cells -- this function's construction order (target cell,
    then Control_Parameter, then direction) satisfies those invariants by
    construction rather than needing to special-case them here.
    """
    target_cells = list(baseline.cells)
    variants: list[ProbeVariant] = []
    for cell_id in target_cells:
        cell_parameters = baseline.cells[cell_id]
        for parameter_name in CONTROL_PARAMETER_NAMES:
            current_value = getattr(cell_parameters, parameter_name)
            for direction in STEP_DIRECTIONS:
                new_value = step_value(parameter_name, current_value, direction)
                if new_value is None:
                    continue
                updated_cells = dict(baseline.cells)
                updated_cells[cell_id] = cell_parameters.model_copy(update={parameter_name: new_value})
                variants.append(
                    ProbeVariant(
                        variant_id=f"{baseline_id}-{cell_id}-{parameter_name}-{direction}",
                        base_parameter_set_id=baseline_id,
                        cell_id=cell_id,
                        parameter_name=parameter_name,
                        direction=direction,
                        parameter_set=ParameterSet(cells=updated_cells),
                    )
                )
    return ProbePlan(baseline_id=baseline_id, baseline=baseline, target_cells=target_cells, variants=variants)


# ---------------------------------------------------------------------------
# Requirement 7.5, 7.6: ordered batching.
# ---------------------------------------------------------------------------


def parameter_set_payloads(plan: ProbePlan) -> list[dict[str, Any]]:
    """Return the Probe_Plan's Parameter_Sets, baseline first, in Probe_Plan order.

    The returned dicts are already in ``mcp_server.predict_batch``'s expected
    ``{"cells": {...}}`` shape.
    """
    ordered = [plan.baseline, *(variant.parameter_set for variant in plan.variants)]
    return [{"cells": parameter_set.model_dump(mode="json")["cells"]} for parameter_set in ordered]


def split_into_batches(
    parameter_sets: list[dict[str, Any]], *, max_batch_size: int = MAX_BATCH_SIZE
) -> list[list[dict[str, Any]]]:
    """Split parameter_sets into order-preserving batches of at most max_batch_size.

    Since ``parameter_set_payloads`` always places the baseline at index 0,
    contiguous slicing here automatically keeps the baseline as the first
    Parameter_Set of the first batch (Requirement 7.6) without any special
    casing.
    """
    if not parameter_sets:
        return []
    return [parameter_sets[start : start + max_batch_size] for start in range(0, len(parameter_sets), max_batch_size)]


# ---------------------------------------------------------------------------
# Requirement 7.7, 7.9, 7.10, 7.11, 7.12: batch execution and percent change.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbePredictionResult:
    """One Parameter_Set's raw Inference_Service prediction (baseline or a variant).

    ``confidence_interval_width`` is optional and defaults to ``None``: today's
    real Inference_Service (``smo/aimlfw/inference_service/schemas.py``'s
    ``PredictionResult``) never returns any confidence-interval field, so this
    is always ``None`` when built from a real prediction via
    ``_to_prediction_result``. It exists purely so
    ``judge_degradation_verdict`` (Requirement 8.4/8.12) can be written
    against a representation that *could* carry one, without requiring a
    change to this dataclass if Inference_Service is extended later. See
    ``judge_degradation_verdict``'s docstring for the full reasoning.
    """

    target_kpi: dict[str, float]
    cell_kpi: dict[str, dict[str, float]]
    confidence_interval_width: dict[str, float] | None = None


@dataclass(frozen=True)
class ProbeVariantResult:
    """One Probe_Variant's prediction plus its per-Target_KPI percent change."""

    variant: ProbeVariant
    prediction: ProbePredictionResult
    percent_change: dict[str, PercentChange]


@dataclass(frozen=True)
class ProbeExecutionResult:
    """The complete result of executing one Probe_Plan against GNN_MCP_Server."""

    plan: ProbePlan
    model_name: str
    model_version: int
    baseline_prediction: ProbePredictionResult
    variant_results: list[ProbeVariantResult] = field(default_factory=list)


def _is_tool_error(result: dict[str, Any]) -> bool:
    """Return whether a GNN_MCP_Server tool result is an error body rather than a success body.

    ``mcp_server.predict_batch``'s success body is a ``BatchPredictionResponse``
    dict (``model_name``/``model_version``/``applied_time_step``/``predictions``);
    its error body is always ``{"error_code", "message", "details"}``
    (smo.aimlfw.common.errors.ErrorResponse's shape). Neither shape overlaps,
    so checking for ``error_code`` is an unambiguous discriminator.
    """
    return "error_code" in result


def _send_batch_with_retries(
    batch: list[dict[str, Any]],
    *,
    batch_index: int,
    time_step: int | None,
    predict_batch: Callable[..., dict[str, Any]],
    max_retries: int,
) -> dict[str, Any]:
    """Send one batch, retrying up to ``max_retries`` additional times on failure (Requirement 7.11)."""
    last_error: dict[str, Any] = {}
    for _attempt in range(max_retries + 1):
        result = predict_batch(parameter_sets=batch, time_step=time_step)
        if not _is_tool_error(result):
            return result
        last_error = result
    raise ProbeExecutionError(
        "probe_batch_failed",
        f"Batch {batch_index} failed after {max_retries + 1} attempt(s)",
        {
            "batch_index": batch_index,
            "attempts": max_retries + 1,
            "underlying_error_code": last_error.get("error_code"),
            "underlying_message": last_error.get("message"),
            "underlying_details": last_error.get("details", {}),
        },
    )


def _to_prediction_result(prediction: dict[str, Any]) -> ProbePredictionResult:
    # "confidence_interval_width" is not part of today's real Inference_Service
    # response shape (smo/aimlfw/inference_service/schemas.py's PredictionResult) --
    # this ``.get`` simply carries one through unrounded if a future revision ever
    # adds it, without requiring any change here. See ProbePredictionResult's docstring.
    raw_ci_width = prediction.get("confidence_interval_width")
    return ProbePredictionResult(
        target_kpi=dict(prediction["target_kpi"]),
        cell_kpi={cell_id: dict(values) for cell_id, values in prediction["cell_kpi"].items()},
        confidence_interval_width=dict(raw_ci_width) if raw_ci_width is not None else None,
    )


def _percent_change(baseline_kpi: dict[str, float], variant_kpi: dict[str, float]) -> dict[str, PercentChange]:
    """Per-Target_KPI percent change, `"undefined"` when the baseline value is exactly 0 (7.7, 7.10)."""
    changes: dict[str, PercentChange] = {}
    for target_kpi, baseline_value in baseline_kpi.items():
        if target_kpi not in variant_kpi:
            continue
        variant_value = variant_kpi[target_kpi]
        if baseline_value == 0:
            changes[target_kpi] = "undefined"
        else:
            changes[target_kpi] = round((variant_value - baseline_value) / abs(baseline_value) * 100, 1)
    return changes


def execute_probe_plan(
    plan: ProbePlan,
    *,
    time_step: int | None = None,
    predict_batch: Callable[..., dict[str, Any]] = mcp_server.predict_batch,
    max_batch_retries: int = DEFAULT_MAX_BATCH_RETRIES,
) -> ProbeExecutionResult:
    """Submit a Probe_Plan's Parameter_Sets to GNN_MCP_Server and compute percent changes.

    - Requirement 7.5/7.6: the plan's baseline + variants are submitted as one
      Batch_Prediction_Request when their total count is <=64, or split into
      ordered <=64-sized batches otherwise (see ``split_into_batches``), sent
      sequentially and always in Probe_Plan order.
    - Requirement 7.9: Inference_Service is reached *only* through the
      ``predict_batch`` callable, which defaults to
      ``mcp_server.predict_batch`` -- the exact same in-process Python tool
      function GNN_MCP_Server's own contract tests call directly. This
      function never opens its own HTTP connection to Inference_Service.
    - Requirement 7.11: each batch gets up to ``max_batch_retries`` additional
      attempts (default 2, so 3 total) on failure; if every attempt for one
      batch fails, the whole execution aborts with that batch's index and
      underlying error reason, and no predictions from *any* batch are
      returned.
    - Requirement 7.12: if batches disagree on ``model_name``/``model_version``,
      the whole execution result is discarded in favor of a
      ``model_version_mismatch`` error.
    - Requirement 7.7/7.10: each Probe_Variant's percent change per Target_KPI
      is ``(variant - baseline) / |baseline| * 100`` rounded to 1 decimal
      place, or the string ``"undefined"`` when the baseline prediction for
      that Target_KPI is exactly 0 (raw baseline/variant values are still
      available via ``ProbeExecutionResult.baseline_prediction``/
      ``ProbeVariantResult.prediction``).
    """
    payloads = parameter_set_payloads(plan)
    batches = split_into_batches(payloads)

    predictions: list[dict[str, Any]] = []
    model_name: str | None = None
    model_version: int | None = None
    for batch_index, batch in enumerate(batches):
        result = _send_batch_with_retries(
            batch,
            batch_index=batch_index,
            time_step=time_step,
            predict_batch=predict_batch,
            max_retries=max_batch_retries,
        )
        batch_model_name, batch_model_version = result["model_name"], result["model_version"]
        if model_name is None:
            model_name, model_version = batch_model_name, batch_model_version
        elif (batch_model_name, batch_model_version) != (model_name, model_version):
            raise ProbeExecutionError(
                "model_version_mismatch",
                "GNN_MCP_Server batches reported different model_name/model_version",
                {
                    "expected_model_name": model_name,
                    "expected_model_version": model_version,
                    "actual_model_name": batch_model_name,
                    "actual_model_version": batch_model_version,
                    "batch_index": batch_index,
                },
            )
        predictions.extend(result["predictions"])

    baseline_prediction = _to_prediction_result(predictions[0])
    variant_results = [
        ProbeVariantResult(
            variant=variant,
            prediction=(variant_prediction := _to_prediction_result(predictions[offset])),
            percent_change=_percent_change(baseline_prediction.target_kpi, variant_prediction.target_kpi),
        )
        for offset, variant in enumerate(plan.variants, start=1)
    ]

    return ProbeExecutionResult(
        plan=plan,
        model_name=model_name,  # type: ignore[arg-type]  # always set: batches is non-empty (>=1 Parameter_Set).
        model_version=model_version,  # type: ignore[arg-type]
        baseline_prediction=baseline_prediction,
        variant_results=variant_results,
    )


# ---------------------------------------------------------------------------
# Requirement 7.1-7.12 end-to-end orchestration for one xApp request.
# ---------------------------------------------------------------------------


def build_and_execute_probe_plan(
    xapp_request: XAppParameterRequest,
    *,
    time_step: int | None = None,
    feature_store: FeatureStore | None = None,
    feature_group: str | None = None,
    predict_batch: Callable[..., dict[str, Any]] = mcp_server.predict_batch,
    max_batch_retries: int = DEFAULT_MAX_BATCH_RETRIES,
) -> ProbeExecutionResult:
    """Baseline -> Probe_Plan -> batch execution for one xApp parameter request.

    Requirement 7.8's ``empty_request`` check is applied here defensively.
    ``schemas.XAppParameterRequest`` (task 7.1) already enforces
    ``target_cells`` ``min_length=1`` and ``parameter_overrides``
    ``min_length=1`` with at least one non-None override per cell via its own
    Pydantic validators, so for any request that reached this function *by
    being validated as an* ``InvokeRequest.xapp_request`` *field*, an empty
    ``target_cells``/``parameter_overrides`` is already unreachable -- this
    check can never fire on that path. It is kept regardless because this
    function is also a valid direct entry point (e.g. from tests, or from a
    future caller that constructs an ``XAppParameterRequest`` without going
    through ``InvokeRequest``), and because it documents the Requirement 7.8
    contract at the layer that actually raises the Probe_Plan.
    """
    if not xapp_request.target_cells or not xapp_request.parameter_overrides:
        raise ProbeExecutionError(
            "empty_request",
            "xApp request specifies no target cells or no Control_Parameter overrides",
        )

    store = feature_store if feature_store is not None else get_feature_store()
    baseline = build_baseline(xapp_request, store, feature_group=feature_group)
    baseline_id = f"baseline-{uuid.uuid4().hex}"
    plan = build_probe_plan(baseline, baseline_id)
    return execute_probe_plan(
        plan,
        time_step=time_step,
        predict_batch=predict_batch,
        max_batch_retries=max_batch_retries,
    )


# ---------------------------------------------------------------------------
# Requirement 8.1, 8.10: Threshold_KPI configuration loading and validation.
# ---------------------------------------------------------------------------


def _threshold_configuration_error(exc: ConfigurationError) -> ProbeExecutionError:
    """Wrap a ``ConfigurationError`` from ``load_kpi_thresholds``/``load_objective_primary_kpi``.

    ``KpiThresholdConfig``'s own "every Target_KPI must have an entry" model
    validator and ``ThresholdKpi``'s own "at least one bound"/"bounds must be
    finite numbers"/"lower_bound cannot exceed upper_bound" field/model
    validators (``smo.aimlfw.common.models``) already perform every check
    Requirement 8.1/8.10 asks for -- this function only needs to translate a
    failure of those checks (surfaced by ``smo.aimlfw.common.config`` as a
    ``ConfigurationError`` wrapping the underlying ``pydantic.ValidationError``)
    into this agent's own stable ``ProbeExecutionError`` shape, best-effort
    extracting *which* Target_KPI failed from the wrapped validation error's
    location path so the error can name it per 8.10's contract.
    """
    target_kpi: str | None = None
    cause = exc.__cause__
    if isinstance(cause, ValidationError):
        for error in cause.errors():
            candidate = next((str(part) for part in error.get("loc", ()) if str(part) in TARGET_KPIS), None)
            if candidate is not None:
                target_kpi = candidate
                break
    return ProbeExecutionError(
        "threshold_configuration_invalid",
        f"Threshold_KPI configuration could not be loaded or validated: {exc}",
        {"target_kpi": target_kpi, "reason": str(exc)},
    )


def load_threshold_config(path: Path | str | None = None) -> KpiThresholdConfig:
    """Load and validate Threshold_KPI definitions (Requirement 8.1), or raise ``ProbeExecutionError`` (8.10)."""
    try:
        return load_kpi_thresholds(path)
    except ConfigurationError as exc:
        raise _threshold_configuration_error(exc) from exc


def load_objective_primary_kpi_config(path: Path | str | None = None) -> ObjectivePrimaryKpiConfig:
    """Load the xApp_Objective -> primary Target_KPI mapping used by Requirement 8.5.

    Raises ``ProbeExecutionError`` if the configuration cannot be loaded or validated.
    """
    try:
        return load_objective_primary_kpi(path)
    except ConfigurationError as exc:
        raise _threshold_configuration_error(exc) from exc


# ---------------------------------------------------------------------------
# Requirement 8.2, 8.3, 8.4, 8.12: Degradation_Verdict judgment.
# ---------------------------------------------------------------------------

ViolationDirection = Literal["below_lower", "above_upper"]
UnknownReasonCode = Literal["confidence_interval_absent", "confidence_interval_too_wide"]

# Requirement 8.3: at most 10 violated Target_KPIs are reported.
MAX_REPORTED_VIOLATIONS = 10
# Requirement 8.4: unknown when confidence-interval width exceeds 50% of the predicted value.
CONFIDENCE_INTERVAL_UNKNOWN_RATIO_THRESHOLD = 0.5


@dataclass(frozen=True)
class ThresholdViolation:
    """One Target_KPI's boundary violation (Requirement 8.3): name, predicted value, threshold, direction."""

    target_kpi: str
    predicted_value: float
    lower_bound: float | None
    upper_bound: float | None
    direction: ViolationDirection


@dataclass(frozen=True)
class UnknownKpiReason:
    """One Target_KPI's reason for contributing to an ``unknown`` Degradation_Verdict (8.4, 8.12)."""

    target_kpi: str
    reason: UnknownReasonCode
    confidence_interval_width_ratio: float | None = None


@dataclass(frozen=True)
class DegradationJudgment:
    """The Degradation_Verdict plus whichever detail list backs it (violations xor unknown reasons)."""

    verdict: Literal["acceptable", "degrading", "unknown"]
    violations: list[ThresholdViolation] = field(default_factory=list)
    unknown_reasons: list[UnknownKpiReason] = field(default_factory=list)


def _threshold_violation(target_kpi: str, value: float, threshold: ThresholdKpi) -> ThresholdViolation | None:
    """Return the boundary violation for one Target_KPI's predicted value, or None if within bounds (inclusive)."""
    if threshold.lower_bound is not None and value < threshold.lower_bound:
        return ThresholdViolation(target_kpi, value, threshold.lower_bound, threshold.upper_bound, "below_lower")
    if threshold.upper_bound is not None and value > threshold.upper_bound:
        return ThresholdViolation(target_kpi, value, threshold.lower_bound, threshold.upper_bound, "above_upper")
    return None


def _has_same_interval_actual_evidence(
    feature_store: FeatureStore, feature_group: str, time_step: int, target_cells: list[str], target_kpi: str
) -> bool:
    """Return whether Feature_Store holds a real measured value for ``target_kpi`` at this Time_Step/cell set.

    Backs Requirement 8.4's "동일 구간의 실측 근거가 없고" clause. Only ever
    consulted from the confidence-interval-present branch below, which (see
    ``judge_degradation_verdict``'s docstring) is unreachable against today's
    actual Inference_Service responses.
    """
    for cell_id in target_cells:
        for record in feature_store.records(feature_group, cell_id=cell_id, time_step=time_step):
            if record.features.get(target_kpi) is not None:
                return True
    return False


def judge_degradation_verdict(
    baseline_prediction: ProbePredictionResult,
    thresholds: KpiThresholdConfig,
    *,
    feature_store: FeatureStore | None = None,
    feature_group: str | None = None,
    time_step: int | None = None,
    target_cells: list[str] | None = None,
) -> DegradationJudgment:
    """Classify the Baseline_Parameter_Set prediction as acceptable/degrading/unknown.

    - Requirement 8.2/8.3: every defined Threshold_KPI's bounds are checked
      inclusive of the boundary values themselves; any violation makes the
      verdict ``degrading`` (checked and reported before the ``unknown``
      branch below, matching 8.3's "위반이 없고"-gated wording for 8.4).
    - Requirement 8.4/8.12 -- **why both are implemented, and why only 8.12
      is reachable today**: ``ProbePredictionResult.confidence_interval_width``
      is an *optional* field this module defines purely for forward
      compatibility/testability; the real Inference_Service
      (``smo/aimlfw/inference_service/schemas.py``'s ``PredictionResult``)
      never populates any such field, so ``confidence_interval_width`` is
      always ``None`` in production today. That means:
        * 8.12's "신뢰 구간 정보가 없으면 unknown, 사유는 정보 부재" branch is
          the one that actually fires against real predictions -- every
          Target_KPI ends up reported with reason
          ``confidence_interval_absent`` whenever there are 0 threshold
          violations.
        * 8.4's "신뢰 구간 폭이 예측값의 50% 초과 AND 동일 구간 실측 근거 없음"
          branch is fully implemented (including the Feature_Store
          same-interval-evidence check via ``_has_same_interval_actual_evidence``)
          so that a future Inference_Service revision that *does* start
          returning confidence intervals is judged correctly without any
          change to this function -- but it can never execute today, since
          its precondition (CI data present) is never true.
    """
    violations: list[ThresholdViolation] = []
    for target_kpi in TARGET_KPIS:
        threshold = thresholds.thresholds[target_kpi]
        value = baseline_prediction.target_kpi[target_kpi]
        violation = _threshold_violation(target_kpi, value, threshold)
        if violation is not None:
            violations.append(violation)
    if violations:
        return DegradationJudgment("degrading", violations[:MAX_REPORTED_VIOLATIONS], [])

    unknown_reasons: list[UnknownKpiReason] = []
    for target_kpi in TARGET_KPIS:
        ci_width = (
            baseline_prediction.confidence_interval_width.get(target_kpi)
            if baseline_prediction.confidence_interval_width is not None
            else None
        )
        if ci_width is None:
            # Requirement 8.12: no confidence-interval information at all.
            unknown_reasons.append(UnknownKpiReason(target_kpi, "confidence_interval_absent"))
            continue
        value = baseline_prediction.target_kpi[target_kpi]
        if value == 0 or abs(ci_width) <= CONFIDENCE_INTERVAL_UNKNOWN_RATIO_THRESHOLD * abs(value):
            continue
        has_actual = (
            feature_store is not None
            and feature_group is not None
            and time_step is not None
            and target_cells is not None
            and _has_same_interval_actual_evidence(feature_store, feature_group, time_step, target_cells, target_kpi)
        )
        if not has_actual:
            # Requirement 8.4: CI width exceeds 50% of the predicted value and no matching actual evidence.
            ratio = abs(ci_width) / abs(value)
            unknown_reasons.append(UnknownKpiReason(target_kpi, "confidence_interval_too_wide", ratio))

    if unknown_reasons:
        return DegradationJudgment("unknown", [], unknown_reasons)
    return DegradationJudgment("acceptable", [], [])


# ---------------------------------------------------------------------------
# Requirement 8.5-8.8, 8.11: recommendation candidacy, ranking, and exclusion.
# ---------------------------------------------------------------------------

ExclusionReason = Literal[
    "threshold_violation",
    "primary_kpi_improvement_below_threshold",
    "missing_prediction",
    "no_objective_specified",
]

# Requirement 8.5: >= 1.0 percent primary-KPI improvement is required for candidacy.
MIN_PRIMARY_IMPROVEMENT_PERCENT = 1.0
# Requirement 8.6: top 5 candidates are returned.
MAX_RECOMMENDATIONS = 5


@dataclass(frozen=True)
class RecommendationCandidate:
    """One Probe_Variant that satisfies both of Requirement 8.5's candidacy conditions."""

    variant: ProbeVariant
    primary_improvement_percent: float


@dataclass(frozen=True)
class ExcludedVariant:
    """One Probe_Variant's Baseline-retention reason for Requirement 8.8."""

    variant_id: str
    reason: ExclusionReason
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecommendationResult:
    """Requirement 8.5-8.8's full candidacy outcome: ranked recommendations plus every exclusion reason."""

    recommendations: list[Recommendation]
    excluded: list[ExcludedVariant]


def _primary_kpi_entry(
    xapp_objective: XAppObjective | None, objective_primary_kpi: ObjectivePrimaryKpiConfig
) -> ObjectivePrimaryKpiEntry | None:
    """Return the objective's primary Target_KPI entry, or None if no objective-driven candidacy applies.

    Requirement 8.5's condition for recommendation candidacy is a *conjunction*:
    "Threshold_KPI 를 만족하고, xApp_Objective 에 대해 ... 주 Target_KPI 가 ...
    개선되면" -- read literally, a candidate must both satisfy every
    Threshold_KPI bound *and* have its objective's primary Target_KPI improve
    by >=1.0%. That second clause presupposes an xApp_Objective exists to
    look a primary Target_KPI up for in the first place (mirroring
    ``ObjectivePrimaryKpiConfig``'s own deliberate exclusion of
    ``"unspecified"`` -- see ``smo.aimlfw.common.config``'s comment on
    ``_OBJECTIVE_DRIVEN_XAPP_OBJECTIVES``). This module therefore treats
    "no xApp_Objective supplied" (``None`` or the literal ``"unspecified"``)
    as "no candidate is possible" rather than inventing a KPI to substitute
    for the missing primary-KPI clause: every Probe_Variant is excluded with
    reason ``no_objective_specified`` in that case (Requirement 8.8), and the
    Baseline_Parameter_Set is retained. Nothing elsewhere in Requirement 7 or
    8 defines an alternate primary KPI to use when no objective is given, so
    this is the only reading that does not require guessing one.
    """
    if xapp_objective is None or xapp_objective == "unspecified":
        return None
    return objective_primary_kpi.objectives[xapp_objective]


def build_recommendations(
    variant_results: list[ProbeVariantResult],
    thresholds: KpiThresholdConfig,
    *,
    xapp_objective: XAppObjective | None,
    objective_primary_kpi: ObjectivePrimaryKpiConfig,
) -> RecommendationResult:
    """Filter Probe_Variants into ranked recommendation candidates (Requirement 8.5-8.8, 8.11).

    Candidacy (8.5) requires, in this exclusion-reason priority order:

    1. every Target_KPI's prediction is present (else ``missing_prediction``, 8.11) --
       this also covers the primary Target_KPI, since it is always one of the
       Threshold_KPI-covered ``TARGET_KPIS``.
    2. every Threshold_KPI bound is satisfied inclusive of the boundary itself
       (else ``threshold_violation``).
    3. an xApp_Objective was supplied (else ``no_objective_specified`` -- see
       ``_primary_kpi_entry``'s docstring).
    4. the objective's primary Target_KPI improved by >=1.0% per its
       improve_direction, using the already-1-decimal-rounded percent change
       from ``execute_probe_plan`` (else ``primary_kpi_improvement_below_threshold``,
       which also covers the ``"undefined"`` percent-change case -- a zero
       baseline value carries no usable improvement evidence either).

    Ranking (8.6/8.7): 2+ candidates are sorted by primary-KPI-improvement
    percent descending, with variant_id ascending as the tiebreak, and the
    top 5 are returned. Because ``execute_probe_plan`` already rounds every
    percent change to one decimal place (Requirement 7.7), "improvement
    differs by <=0.1 percentage points" collapses to "improvement is equal
    after rounding" for any two already-rounded values one decimal place
    apart -- so a plain ``(-improvement, variant_id)`` sort key satisfies
    8.6's tiebreak without needing a separate fuzzy-equality clustering pass.
    Exactly 1 candidate (8.7) and 0 candidates (8.8) fall out of the same
    general sort/slice with no special-casing.
    """
    primary_entry = _primary_kpi_entry(xapp_objective, objective_primary_kpi)
    no_objective = primary_entry is None

    candidates: list[RecommendationCandidate] = []
    excluded: list[ExcludedVariant] = []

    for variant_result in variant_results:
        variant = variant_result.variant
        target_kpi_values = variant_result.prediction.target_kpi

        missing = sorted(name for name in TARGET_KPIS if name not in target_kpi_values)
        if missing:
            excluded.append(
                ExcludedVariant(variant.variant_id, "missing_prediction", {"missing_target_kpis": missing})
            )
            continue

        violation = next(
            (
                violation
                for target_kpi in TARGET_KPIS
                if (
                    violation := _threshold_violation(
                        target_kpi, target_kpi_values[target_kpi], thresholds.thresholds[target_kpi]
                    )
                )
                is not None
            ),
            None,
        )
        if violation is not None:
            excluded.append(
                ExcludedVariant(
                    variant.variant_id,
                    "threshold_violation",
                    {"target_kpi": violation.target_kpi, "direction": violation.direction},
                )
            )
            continue

        if no_objective:
            excluded.append(ExcludedVariant(variant.variant_id, "no_objective_specified", {}))
            continue

        raw_change = variant_result.percent_change.get(primary_entry.target_kpi)
        if raw_change is None or raw_change == "undefined":
            excluded.append(
                ExcludedVariant(
                    variant.variant_id,
                    "missing_prediction",
                    {"missing_target_kpis": [primary_entry.target_kpi], "reason": "baseline_value_is_zero"},
                )
            )
            continue

        signed_improvement = raw_change if primary_entry.improve_direction == "higher_is_better" else -raw_change
        if signed_improvement < MIN_PRIMARY_IMPROVEMENT_PERCENT:
            excluded.append(
                ExcludedVariant(
                    variant.variant_id,
                    "primary_kpi_improvement_below_threshold",
                    {"target_kpi": primary_entry.target_kpi, "improvement_percent": signed_improvement},
                )
            )
            continue

        candidates.append(RecommendationCandidate(variant, signed_improvement))

    def _sort_key(candidate: RecommendationCandidate) -> tuple[float, str]:
        return (-candidate.primary_improvement_percent, candidate.variant.variant_id)

    ordered = sorted(candidates, key=_sort_key)
    top = ordered[:MAX_RECOMMENDATIONS]
    recommendations = [
        Recommendation(
            variant_id=candidate.variant.variant_id,
            parameter_set=candidate.variant.parameter_set,
            improvement_percent=candidate.primary_improvement_percent,
        )
        for candidate in top
    ]
    return RecommendationResult(recommendations, excluded)


# ---------------------------------------------------------------------------
# Requirement 8.9, 12.6-12.8: assembling the single InvokeResponse.
#
# Evidence is assembled and persisted before the public response is created.
# ---------------------------------------------------------------------------


def build_marginal_effects(probe_result: ProbeExecutionResult) -> list[MarginalEffectRecord]:
    """Derive an increasing-parameter marginal effect from +/- one-step probes."""
    by_parameter: dict[tuple[str, str], dict[str, ProbeVariantResult]] = {}
    for result in probe_result.variant_results:
        key = (result.variant.cell_id, result.variant.parameter_name)
        by_parameter.setdefault(key, {})[result.variant.direction] = result

    effects: list[MarginalEffectRecord] = []
    for (cell_id, parameter_name), directions in sorted(by_parameter.items()):
        for target_kpi in TARGET_KPIS:
            plus = directions.get("+1")
            minus = directions.get("-1")
            plus_value = None if plus is None else plus.percent_change.get(target_kpi)
            minus_value = None if minus is None else minus.percent_change.get(target_kpi)
            numeric_plus = plus_value if isinstance(plus_value, float) else None
            numeric_minus = minus_value if isinstance(minus_value, float) else None
            if numeric_plus is not None and numeric_minus is not None:
                value = (numeric_plus - numeric_minus) / 2.0
            elif numeric_plus is not None:
                value = numeric_plus
            elif numeric_minus is not None:
                value = -numeric_minus
            else:
                value = None
            effects.append(
                MarginalEffectRecord(
                    cell_id=cell_id,
                    control_parameter=parameter_name,
                    target_kpi=target_kpi,
                    value_percent=None if value is None else round(max(-100.0, min(100.0, value)), 1),
                )
            )
    return effects


_EXCLUSION_REASON_LABELS_KO: dict[ExclusionReason, str] = {
    "threshold_violation": "Threshold_KPI 위반",
    "primary_kpi_improvement_below_threshold": "주 Target_KPI 개선폭 1.0% 미달",
    "missing_prediction": "예측값 누락",
    "no_objective_specified": "xApp_Objective 미지정",
}
_EXCLUSION_REASON_LABELS_EN: dict[ExclusionReason, str] = {
    "threshold_violation": "Threshold_KPI violation",
    "primary_kpi_improvement_below_threshold": "primary Target_KPI improvement below 1.0%",
    "missing_prediction": "missing prediction",
    "no_objective_specified": "no xApp_Objective specified",
}


def _exclusion_reason_counts(excluded: list[ExcludedVariant]) -> dict[ExclusionReason, int]:
    counts: dict[ExclusionReason, int] = {}
    for item in excluded:
        counts[item.reason] = counts.get(item.reason, 0) + 1
    return counts


def _build_rationale_summary(
    *,
    korean: bool,
    judgment: DegradationJudgment,
    recommendations: list[Recommendation],
    excluded: list[ExcludedVariant],
) -> str:
    """Compose the single 1..2000-character rationale text for Requirement 8.9/12.6.

    Requirement 8.8's per-Probe_Variant exclusion reasons and 8.3's
    per-Target_KPI violation detail are summarized here rather than
    enumerated one-by-one, because ``schemas.InvokeResponse`` (task 7.1's
    fixed public contract) has no field for either -- ``rationale_summary``
    (capped at 2000 characters) is the only text channel available for this
    response, and up to 16 target cells x 5 parameters x 2 directions can
    produce up to 160 Probe_Variants, so a literal per-variant listing could
    overflow the 2000-character cap. Counting by exclusion-reason category
    keeps the summary bounded regardless of Probe_Plan size while still
    surfacing every distinct reason category from 8.8. The complete,
    unsummarized detail (every violation and every exclusion, structured) is
    still available to callers of ``judge_degradation_verdict``/
    ``build_recommendations`` directly and is persisted in the Evidence_Record.
    """
    if korean:
        if judgment.verdict == "degrading":
            names = ", ".join(violation.target_kpi for violation in judgment.violations)
            parts = [
                f"기준 파라미터 조합은 {len(judgment.violations)}개의 Target_KPI({names})에서 "
                "임계값을 위반하여 열화(degrading)로 판정되었습니다."
            ]
        elif judgment.verdict == "unknown":
            names = ", ".join(reason.target_kpi for reason in judgment.unknown_reasons)
            parts = [f"일부 Target_KPI({names})의 신뢰 구간 근거가 부족하여 판정을 unknown으로 설정했습니다."]
        else:
            parts = ["기준 파라미터 조합은 모든 Threshold_KPI 기준을 충족하여 acceptable로 판정되었습니다."]

        if recommendations:
            parts.append(f"추천 대안 {len(recommendations)}개를 개선폭 내림차순으로 제시합니다.")
        else:
            counts = _exclusion_reason_counts(excluded)
            if counts:
                detail = ", ".join(
                    f"{_EXCLUSION_REASON_LABELS_KO[reason]} {count}건" for reason, count in counts.items()
                )
                parts.append(f"추천 가능한 대안이 없어 기준 조합을 유지합니다 (제외 사유: {detail}).")
            else:
                parts.append("추천 가능한 대안이 없어 기준 조합을 유지합니다.")
        parts.append("파라미터 변경은 Policy_Manager 의 A1 정책 발행 경로를 통해서만 적용됩니다.")
        summary = " ".join(parts)
    else:
        if judgment.verdict == "degrading":
            names = ", ".join(violation.target_kpi for violation in judgment.violations)
            parts = [
                f"The baseline parameter combination violates thresholds on {len(judgment.violations)} "
                f"Target_KPI(s) ({names}) and is judged degrading."
            ]
        elif judgment.verdict == "unknown":
            names = ", ".join(reason.target_kpi for reason in judgment.unknown_reasons)
            parts = [
                f"Confidence-interval evidence is insufficient for Target_KPI(s) ({names}); "
                "the verdict is unknown."
            ]
        else:
            parts = ["The baseline parameter combination satisfies every Threshold_KPI and is judged acceptable."]

        if recommendations:
            parts.append(
                f"{len(recommendations)} recommended alternative(s) are provided in descending improvement order."
            )
        else:
            counts = _exclusion_reason_counts(excluded)
            if counts:
                detail = ", ".join(
                    f"{_EXCLUSION_REASON_LABELS_EN[reason]}: {count}" for reason, count in counts.items()
                )
                parts.append(
                    f"No recommendation candidates remain; the baseline is retained (exclusion reasons: {detail})."
                )
            else:
                parts.append("No recommendation candidates remain; the baseline is retained.")
        parts.append("Parameter changes are applied exclusively through the Policy_Manager's A1 publish path.")
        summary = " ".join(parts)

    return summary[:2000]


# ---------------------------------------------------------------------------
# Requirement 7.1-7.12, 8.1-8.12, 12.6-12.8 end-to-end orchestration.
# ---------------------------------------------------------------------------


async def run_advisor(request: InvokeRequest) -> InvokeResponse:
    """Execute one ``POST /invoke`` request end-to-end and return its atomic InvokeResponse.

        request.baseline_parameter_set -> build_probe_plan -> execute_probe_plan --\\
        request.xapp_request ---------> build_and_execute_probe_plan -------------> judge_degradation_verdict
                                                                                     -> build_recommendations
                                                                                     -> InvokeResponse

    Requirement 7.9/12.6-12.8: nothing here calls Policy_Manager or any A1
    mutation path -- the returned recommendations are advisory data only.

    Raises ``AdvisorNotReadyError`` if no model is currently loaded/ready
    (mirrors ``get_model_health``'s own readiness check, so an unready
    advisor fails fast before attempting any GNN_MCP_Server call).
    ``ProbeExecutionError`` (a plain ``Exception`` subclass) propagates for
    every Requirement 7/8 failure mode (``empty_request``,
    ``feature_record_unavailable``, ``probe_batch_failed``,
    ``model_version_mismatch``, ``threshold_configuration_invalid``);
    ``server.py``'s existing catch-all ``except Exception`` handler already
    turns any of these into a ``502 advisor_error`` response carrying that
    exception's message, so no server.py change is required to surface them.
    """
    if get_model_health().status != "ready":
        raise AdvisorNotReadyError("KPI Advisor has no ready model to serve predictions")

    thresholds = load_threshold_config()
    objective_primary_kpi = load_objective_primary_kpi_config()

    if request.baseline_parameter_set is not None:
        baseline_id = f"baseline-{uuid.uuid4().hex}"
        plan = build_probe_plan(request.baseline_parameter_set, baseline_id)
        target_cells = list(request.baseline_parameter_set.cells)
        xapp_objective: XAppObjective | None = None
        probe_result = execute_probe_plan(plan, time_step=request.time_step)
    else:
        xapp_request = request.xapp_request
        assert xapp_request is not None  # InvokeRequest.require_one_parameter_source guarantees exactly one is set.
        target_cells = xapp_request.target_cells
        xapp_objective = xapp_request.xapp_objective
        probe_result = build_and_execute_probe_plan(xapp_request, time_step=request.time_step)

    judgment = judge_degradation_verdict(
        probe_result.baseline_prediction,
        thresholds,
        feature_store=get_feature_store(),
        feature_group=_feature_group(),
        time_step=request.time_step,
        target_cells=target_cells,
    )
    recommendation_result = build_recommendations(
        probe_result.variant_results,
        thresholds,
        xapp_objective=xapp_objective,
        objective_primary_kpi=objective_primary_kpi,
    )
    rationale_summary = _build_rationale_summary(
        korean=intent_is_korean(request.intent),
        judgment=judgment,
        recommendations=recommendation_result.recommendations,
        excluded=recommendation_result.excluded,
    )
    marginal_effects = build_marginal_effects(probe_result)
    conflict_result = analyze(marginal_effects, load_runtime_thresholds().conflict_threshold)
    evidence = assemble_evidence_record(
        probe_result=probe_result,
        thresholds=thresholds,
        judgment=judgment,
        recommendation_result=recommendation_result,
        marginal_effects=marginal_effects,
        conflict_result=conflict_result,
    )
    stored_evidence = persist_evidence(evidence)
    return InvokeResponse(
        degradation_verdict=stored_evidence.degradation_verdict,
        recommendations=[Recommendation.model_validate(item) for item in stored_evidence.recommendations],
        evidence_record_id=stored_evidence.evidence_id,
        rationale_summary=rationale_summary,
    )
