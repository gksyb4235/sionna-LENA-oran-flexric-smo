"""Task 8.8 sanity-check tests for Temporal_Scheduler (design.md 10, Requirement 10).

Lightweight, non-exhaustive unit coverage (Hypothesis property tests for
Requirement 10 are separate future tasks -- 8.9 through 8.17). Follows
``test_agent_baseline_and_probe_plan.py``'s manual ``sys.path`` setup
convention and its fake/injectable Feature_Store pattern (duck-typed against
``FeatureStore.records``), plus a stub ``predict_batch`` callable identical
in spirit to ``test_mcp_server_contract.py``'s stub-transport approach, but
here stubbing the in-process ``predict_batch`` callable ``build_plan`` itself
takes (Requirement 7.9's "GNN_MCP_Server only" constraint applies the same
way to Temporal_Scheduler).

Covers:
- Requirement 10.1/10.4/10.10: basic 5-entry Temporal_Plan construction with
  a single xApp_Objective, picking the best primary-KPI candidate.
- Requirement 10.2: objective-unspecified mode picks the fewest-Threshold_KPI-
  violation candidate and labels it "unspecified".
- Requirement 10.5: sum vs mean aggregation across the 5 Time_Steps.
- Requirement 10.7: transition list computation across an objective switch.
- Requirement 10.8/10.9: Feature_Store candidate fallback and the
  ``no_candidate`` error when Feature_Store has nothing usable either.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

AGENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(AGENT_DIR))

from smo.aimlfw.common.config import (  # noqa: E402
    load_kpi_thresholds,
    load_objective_primary_kpi,
    load_target_kpi_aggregation,
)
from smo.aimlfw.common.constants import TARGET_KPIS  # noqa: E402
from smo.aimlfw.common.models import CellParameters, FeatureRecord, ParameterSet, SampleCount  # noqa: E402

import temporal_scheduler  # noqa: E402

THRESHOLDS = load_kpi_thresholds()
OBJECTIVE_PRIMARY_KPI = load_objective_primary_kpi()
TARGET_KPI_AGGREGATION = load_target_kpi_aggregation()


def _default_kpis(**overrides: float) -> dict[str, float]:
    base = {
        "cell_goodput_mbps": 10.0,
        "avg_ue_goodput_mbps": 10.0,
        "ue_goodput_p5_mbps": 5.0,
        "sinr_p50_db": 10.0,
        "prb_utilization_pct": 50.0,
        "delay_p95_ms": 10.0,
        "ho_failure_count": 0.0,
        "pingpong_count": 0.0,
        "rlf_count": 0.0,
        "interval_energy_j": 50.0,
    }
    base.update(overrides)
    assert set(base) == set(TARGET_KPIS)
    return base


def _cell_parameters(**overrides: Any) -> CellParameters:
    values = {"tx_power_dbm": 40.0, "ret_tilt_deg": 5.0, "cio_bias_db": 0.0, "hysteresis_db": 2.0, "ttt_ms": 160}
    values.update(overrides)
    return CellParameters.model_validate(values)


def _parameter_set(cell_id: str = "gNB_5G", **overrides: Any) -> ParameterSet:
    return ParameterSet(cells={cell_id: _cell_parameters(**overrides)})


def _make_predict_batch(kpi_fn):
    """Build a stub ``predict_batch`` callable matching ``mcp_server.predict_batch``'s response shape."""

    def predict_batch(parameter_sets: list[dict[str, Any]], time_step: int | None = None) -> dict[str, Any]:
        predictions = []
        for index, parameter_set in enumerate(parameter_sets):
            cells = parameter_set["cells"]
            target_kpi = kpi_fn(cells, time_step)
            predictions.append(
                {"index": index, "target_kpi": target_kpi, "cell_kpi": {cell_id: dict(target_kpi) for cell_id in cells}}
            )
        return {
            "model_name": "stub-gnn",
            "model_version": 1,
            "applied_time_step": time_step or 0,
            "predictions": predictions,
        }

    return predict_batch


class _FakeFeatureStore:
    """Duck-typed Feature_Store stand-in: only implements ``.records(...)``."""

    def __init__(self, records: list[FeatureRecord]) -> None:
        self._records = records

    def records(
        self, feature_group: str, *, cell_id: str | None = None, time_step: int | None = None
    ) -> list[FeatureRecord]:
        return [
            record
            for record in self._records
            if record.feature_group == feature_group
            and (cell_id is None or record.cell_id == cell_id)
            and (time_step is None or record.time_step == time_step)
        ]


def _build(**overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "thresholds": THRESHOLDS,
        "objective_primary_kpi": OBJECTIVE_PRIMARY_KPI,
        "target_kpi_aggregation": TARGET_KPI_AGGREGATION,
    }
    kwargs.update(overrides)
    return temporal_scheduler.build_plan(**kwargs)


# ---------------------------------------------------------------------------
# Requirement 10.1/10.4/10.10: basic 5-entry plan with an objective.
# ---------------------------------------------------------------------------


def test_build_plan_objective_driven_picks_best_primary_kpi_candidate() -> None:
    low_power = _parameter_set(tx_power_dbm=30.0)
    high_power = _parameter_set(tx_power_dbm=46.0)

    def kpi_fn(cells: dict[str, Any], _time_step: int | None) -> dict[str, float]:
        tx_power = cells["gNB_5G"]["tx_power_dbm"]
        return _default_kpis(cell_goodput_mbps=tx_power)

    plan = _build(
        objectives=["throughput_maximization"],
        candidates_by_objective={"throughput_maximization": [low_power, high_power]},
        predict_batch=_make_predict_batch(kpi_fn),
    )

    assert [assignment.time_step for assignment in plan.assignments] == [0, 1, 2, 3, 4]
    for assignment in plan.assignments:
        assert assignment.xapp_objective == "throughput_maximization"
        assert assignment.parameter_set.cells["gNB_5G"].tx_power_dbm == 46.0  # higher goodput wins.
        assert assignment.model_name == "stub-gnn"
        assert assignment.model_version == 1
        assert assignment.predicted_kpis["cell_goodput_mbps"]["gNB_5G"] == 46.0
        assert assignment.threshold_violations == []  # 46.0 satisfies the lower_bound=0.0 threshold.


# ---------------------------------------------------------------------------
# Requirement 10.2: objective-unspecified mode picks fewest violations.
# ---------------------------------------------------------------------------


def test_build_plan_unspecified_objective_selects_fewest_violations() -> None:
    good = _parameter_set(tx_power_dbm=30.0)
    bad = _parameter_set(tx_power_dbm=46.0)

    def kpi_fn(cells: dict[str, Any], _time_step: int | None) -> dict[str, float]:
        tx_power = cells["gNB_5G"]["tx_power_dbm"]
        if tx_power == 46.0:
            # Violates cell_goodput_mbps's lower_bound=0.0 and ho_failure_count's lower_bound=0.0.
            return _default_kpis(cell_goodput_mbps=-5.0, ho_failure_count=-1.0)
        return _default_kpis()

    plan = _build(
        objectives=[],
        candidates_by_objective={"pool": [good, bad]},
        predict_batch=_make_predict_batch(kpi_fn),
    )

    for assignment in plan.assignments:
        assert assignment.xapp_objective == "unspecified"
        assert assignment.parameter_set.cells["gNB_5G"].tx_power_dbm == 30.0
        assert assignment.threshold_violations == []


# ---------------------------------------------------------------------------
# Requirement 10.5: sum vs mean aggregation.
# ---------------------------------------------------------------------------


def test_build_plan_aggregates_sum_and_mean_correctly() -> None:
    candidate = _parameter_set()

    def kpi_fn(_cells: dict[str, Any], time_step: int | None) -> dict[str, float]:
        step = time_step or 0
        # cell_goodput_mbps aggregates by sum; sinr_p50_db aggregates by mean (target_kpi_aggregation.json).
        return _default_kpis(cell_goodput_mbps=10.0 + step, sinr_p50_db=5.0 + step)

    plan = _build(
        objectives=["throughput_maximization"],
        candidates_by_objective={"throughput_maximization": [candidate]},
        predict_batch=_make_predict_batch(kpi_fn),
    )

    assert plan.aggregate_kpis["cell_goodput_mbps"] == pytest.approx(10 + 11 + 12 + 13 + 14)
    assert plan.aggregate_kpis["sinr_p50_db"] == pytest.approx((5 + 6 + 7 + 8 + 9) / 5)


# ---------------------------------------------------------------------------
# Requirement 10.7: transition list.
# ---------------------------------------------------------------------------


def test_build_plan_transition_list_marks_objective_changes() -> None:
    throughput_candidate = _parameter_set(ret_tilt_deg=1.0)
    energy_candidate = _parameter_set(ret_tilt_deg=2.0)

    def kpi_fn(cells: dict[str, Any], time_step: int | None) -> dict[str, float]:
        ret_tilt = cells["gNB_5G"]["ret_tilt_deg"]
        step = time_step or 0
        if ret_tilt == 1.0:
            return _default_kpis(cell_goodput_mbps=100.0)
        # interval_energy_j is lower_is_better; make it very attractive only from Time_Step 2 onward
        # so the winning objective switches partway through (Requirement 10.7).
        return _default_kpis(interval_energy_j=(-500.0 if step >= 2 else 100.0))

    plan = _build(
        objectives=["throughput_maximization", "energy_saving"],
        candidates_by_objective={
            "throughput_maximization": [throughput_candidate],
            "energy_saving": [energy_candidate],
        },
        predict_batch=_make_predict_batch(kpi_fn),
    )

    objectives_by_step = [assignment.xapp_objective for assignment in plan.assignments]
    assert objectives_by_step == [
        "throughput_maximization",
        "throughput_maximization",
        "energy_saving",
        "energy_saving",
        "energy_saving",
    ]
    assert plan.transition_time_steps == [2]


# ---------------------------------------------------------------------------
# Requirement 10.8/10.9: Feature_Store candidate fallback and no_candidate.
# ---------------------------------------------------------------------------


def _feature_record(*, cell_id: str, time_step: int, data_quality: str = "complete") -> FeatureRecord:
    return FeatureRecord(
        feature_group="default",
        time_step=time_step,
        cell_id=cell_id,
        control_parameters=_cell_parameters(),
        features={"cell_goodput_mbps": 1.0},
        sample_counts={"cell_goodput_mbps": SampleCount(valid=1, excluded=0)},
        data_quality=data_quality,
        source_dir="/tmp/results/example",
        seed=0,
    )


def test_build_plan_derives_candidate_from_feature_store_when_none_given() -> None:
    store = _FakeFeatureStore(
        [_feature_record(cell_id="gNB_5G", time_step=step) for step in range(5)]
    )

    plan = _build(
        objectives=[],
        candidates_by_objective={},
        feature_store=store,
        predict_batch=_make_predict_batch(lambda _cells, _ts: _default_kpis()),
    )

    assert len(plan.assignments) == 5
    for assignment in plan.assignments:
        assert assignment.xapp_objective == "unspecified"
        assert assignment.parameter_set.cells["gNB_5G"].tx_power_dbm == 40.0


def test_build_plan_raises_no_candidate_when_feature_store_is_empty_too() -> None:
    store = _FakeFeatureStore([])

    with pytest.raises(temporal_scheduler.TemporalSchedulerError) as excinfo:
        _build(
            objectives=[],
            candidates_by_objective={},
            feature_store=store,
            predict_batch=_make_predict_batch(lambda _cells, _ts: _default_kpis()),
        )
    assert excinfo.value.error_code == "no_candidate"


# ---------------------------------------------------------------------------
# Requirement 10.13: prediction_unavailable, no partial plan.
# ---------------------------------------------------------------------------


def test_build_plan_raises_prediction_unavailable_on_tool_error() -> None:
    def failing_predict_batch(parameter_sets: list[dict[str, Any]], time_step: int | None = None) -> dict[str, Any]:
        return {"error_code": "inference_unavailable", "message": "boom", "details": {}}

    with pytest.raises(temporal_scheduler.TemporalSchedulerError) as excinfo:
        _build(
            objectives=["throughput_maximization"],
            candidates_by_objective={"throughput_maximization": [_parameter_set()]},
            predict_batch=failing_predict_batch,
        )
    assert excinfo.value.error_code == "prediction_unavailable"
