"""Task 7.2 unit tests: Baseline_Parameter_Set construction, Probe_Plan
generation, and batched GNN_MCP_Server execution.

Follows ``test_contract.py``/``test_mcp_server_contract.py``'s manual
``sys.path`` setup convention since ``kpi-advisor`` is not installed as a
package during tests.

Task 7.2 has no dedicated Property test of its own (Properties 41-47 for
Requirement 7 belong to the separate tasks 7.4-7.10), so this module is
plain focused unit-test coverage rather than a Hypothesis property module,
using a fake/injectable Feature_Store (duck-typed against
``FeatureStore.records``) instead of real files or MongoDB.

Requirement 7.1 (Baseline_Parameter_Set 구성):
    WHEN xApp 파라미터 요청이 도착하면, THE KPI_Advisor_Agent SHALL 요청에
    지정되지 않은 Control_Parameter 를 Feature_Store 의 가장 최근
    Feature_Record 값으로 채워 5개 Control_Parameter 가 모두 지정된
    Baseline_Parameter_Set 1개를 구성한다.

Requirement 7.2/7.3 (Probe_Plan 생성과 variant 불변식):
    WHEN Baseline_Parameter_Set 이 확정되면, THE KPI_Advisor_Agent SHALL 각
    Control_Parameter 의 스텝 크기를 사용하여 Probe_Plan 을 생성하고, 각 대상
    셀의 각 Control_Parameter 에 대해 +1/-1 스텝 Probe_Variant 를 포함시키며,
    허용 범위를 벗어나는 방향은 생성하지 않는다.

Requirement 7.4 (전체 방향이 범위 밖이면 0 variant):
    IF 모든 방향이 허용 범위를 벗어나면, THEN Baseline_Parameter_Set 만 담은
    Probe_Plan 을 생성한다.

Requirement 7.5/7.6 (배치 분할과 순서 보존):
    64개 이하이면 단일 배치, 64개 초과이면 baseline-first 순서 보존 분할.

Requirement 7.7/7.10 (변화율 계산과 0 baseline 처리):
    변화율은 소수 첫째 자리까지 반올림하고, baseline 예측값이 0이면
    `"undefined"` 로 표기한다.

Requirement 7.8 (빈 요청 거부):
    Control_Parameter 값도 대상 셀도 없는 요청은 `empty_request` 오류.

Requirement 7.11 (배치 재전송과 실패 중단):
    실패한 배치는 최대 2회 재전송(총 3회 시도)하고, 모두 실패하면 전체 실행을
    중단한다.

Requirement 7.12 (배치 간 모델 버전 일치 검증):
    배치 간 model_name/model_version 이 다르면 전체 결과를 폐기한다.
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

from smo.aimlfw.common.constants import CONTROL_PARAMETER_NAMES, TTT_ALLOWED_MS  # noqa: E402
from smo.aimlfw.common.models import CellParameters, FeatureRecord, SampleCount  # noqa: E402
from smo.aimlfw.inference_service.schemas import MAX_BATCH_SIZE  # noqa: E402

import agent  # noqa: E402
from schemas import PartialCellParameters, XAppParameterRequest  # noqa: E402


def _feature_record(
    *, cell_id: str, time_step: int, control_parameters: dict[str, float | int]
) -> FeatureRecord:
    kpi_names = ("cell_goodput_mbps",)
    return FeatureRecord(
        feature_group="default",
        time_step=time_step,
        cell_id=cell_id,
        control_parameters=CellParameters.model_validate(control_parameters),
        features={name: 1.0 for name in kpi_names},
        sample_counts={name: SampleCount(valid=1, excluded=0) for name in kpi_names},
        data_quality="complete",
        source_dir="/tmp/results/example",
        seed=0,
    )


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


_BASE_CONTROL_PARAMETERS: dict[str, float | int] = {
    "tx_power_dbm": 40.0,
    "ret_tilt_deg": 5.0,
    "cio_bias_db": 0.0,
    "hysteresis_db": 2.0,
    "ttt_ms": 160,
}


def _cell_parameters(**overrides: float | int) -> dict[str, float | int]:
    values = dict(_BASE_CONTROL_PARAMETERS)
    values.update(overrides)
    return values


# ---------------------------------------------------------------------------
# Requirement 7.1: Baseline_Parameter_Set construction.
# ---------------------------------------------------------------------------


def test_build_baseline_fills_unspecified_parameters_from_most_recent_feature_record() -> None:
    store = _FakeFeatureStore(
        [
            _feature_record(cell_id="gNB_5G", time_step=0, control_parameters=_cell_parameters(tx_power_dbm=30.0)),
            # time_step=2 is the most recent record for gNB_5G; its values should win.
            _feature_record(cell_id="gNB_5G", time_step=2, control_parameters=_cell_parameters(tx_power_dbm=44.0)),
            _feature_record(cell_id="gNB_5G", time_step=1, control_parameters=_cell_parameters(tx_power_dbm=38.0)),
        ]
    )
    xapp_request = XAppParameterRequest(
        target_cells=["gNB_5G"],
        parameter_overrides={"gNB_5G": PartialCellParameters(ret_tilt_deg=9.0)},
    )

    baseline = agent.build_baseline(xapp_request, store, feature_group="default")

    cell = baseline.cells["gNB_5G"]
    assert cell.ret_tilt_deg == 9.0  # explicit override wins.
    assert cell.tx_power_dbm == 44.0  # filled from time_step=2 (the most recent record).
    assert cell.cio_bias_db == _BASE_CONTROL_PARAMETERS["cio_bias_db"]
    assert cell.hysteresis_db == _BASE_CONTROL_PARAMETERS["hysteresis_db"]
    assert cell.ttt_ms == _BASE_CONTROL_PARAMETERS["ttt_ms"]


def test_build_baseline_covers_every_target_cell_with_all_five_parameters() -> None:
    store = _FakeFeatureStore(
        [
            _feature_record(cell_id=cell_id, time_step=0, control_parameters=_cell_parameters())
            for cell_id in ("gNB_5G", "gNB_4G_1")
        ]
    )
    xapp_request = XAppParameterRequest(
        target_cells=["gNB_5G", "gNB_4G_1"],
        parameter_overrides={"gNB_5G": PartialCellParameters(tx_power_dbm=41.0)},
    )

    baseline = agent.build_baseline(xapp_request, store, feature_group="default")

    assert set(baseline.cells) == {"gNB_5G", "gNB_4G_1"}
    for cell in baseline.cells.values():
        for name in CONTROL_PARAMETER_NAMES:
            assert getattr(cell, name) is not None
    assert baseline.cells["gNB_5G"].tx_power_dbm == 41.0


def test_build_baseline_raises_probe_execution_error_when_no_feature_record_available() -> None:
    store = _FakeFeatureStore([])
    xapp_request = XAppParameterRequest(
        target_cells=["gNB_5G"],
        parameter_overrides={"gNB_5G": PartialCellParameters(tx_power_dbm=41.0)},
    )

    with pytest.raises(agent.ProbeExecutionError) as excinfo:
        agent.build_baseline(xapp_request, store, feature_group="default")
    assert excinfo.value.error_code == "feature_record_unavailable"


# ---------------------------------------------------------------------------
# Requirement 7.2/7.3: Probe_Plan variant generation invariants.
# ---------------------------------------------------------------------------


def test_build_probe_plan_generates_one_plus_and_one_minus_variant_per_cell_and_parameter() -> None:
    from smo.aimlfw.common.models import ParameterSet

    baseline = ParameterSet(cells={"gNB_5G": CellParameters.model_validate(_cell_parameters())})

    plan = agent.build_probe_plan(baseline, "baseline-1")

    # Every Control_Parameter here is strictly interior to its allowed range/
    # allowed-value set, so both directions must be present for every one of
    # the 5 Control_Parameters: exactly 10 variants for 1 target cell.
    assert len(plan.variants) == 2 * len(CONTROL_PARAMETER_NAMES)
    seen_keys = set()
    for variant in plan.variants:
        assert variant.base_parameter_set_id == "baseline-1"
        assert variant.cell_id == "gNB_5G"
        key = (variant.cell_id, variant.parameter_name, variant.direction)
        assert key not in seen_keys
        seen_keys.add(key)

        # Exactly one (cell_id, Control_Parameter) value differs from baseline.
        baseline_cell = baseline.cells[variant.cell_id]
        variant_cell = variant.parameter_set.cells[variant.cell_id]
        differing = [
            name
            for name in CONTROL_PARAMETER_NAMES
            if getattr(baseline_cell, name) != getattr(variant_cell, name)
        ]
        assert differing == [variant.parameter_name]
        for other_cell_id in baseline.cells:
            if other_cell_id != variant.cell_id:
                assert variant.parameter_set.cells[other_cell_id] == baseline.cells[other_cell_id]

    expected_keys = {
        (cell_id, name, direction)
        for cell_id in baseline.cells
        for name in CONTROL_PARAMETER_NAMES
        for direction in ("+1", "-1")
    }
    assert seen_keys == expected_keys


def test_build_probe_plan_skips_out_of_range_direction() -> None:
    from smo.aimlfw.common.models import ParameterSet

    # tx_power_dbm at the maximum (46) cannot step +1; ttt_ms at the maximum
    # allowed value cannot step +1 either.
    boundary_cell = _cell_parameters(tx_power_dbm=46.0, ttt_ms=TTT_ALLOWED_MS[-1])
    baseline = ParameterSet(cells={"gNB_5G": CellParameters.model_validate(boundary_cell)})

    plan = agent.build_probe_plan(baseline, "baseline-1")

    directions_by_parameter = {
        name: {variant.direction for variant in plan.variants if variant.parameter_name == name}
        for name in CONTROL_PARAMETER_NAMES
    }
    assert directions_by_parameter["tx_power_dbm"] == {"-1"}
    assert directions_by_parameter["ttt_ms"] == {"-1"}
    for name in ("ret_tilt_deg", "cio_bias_db", "hysteresis_db"):
        assert directions_by_parameter[name] == {"+1", "-1"}


# ---------------------------------------------------------------------------
# Requirement 7.4: every direction out of range -> baseline-only Probe_Plan.
# ---------------------------------------------------------------------------


def test_build_probe_plan_returns_zero_variants_when_every_direction_is_out_of_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from smo.aimlfw.common.models import ParameterSet

    monkeypatch.setattr(agent, "step_value", lambda *_args, **_kwargs: None)
    baseline = ParameterSet(cells={"gNB_5G": CellParameters.model_validate(_cell_parameters())})

    plan = agent.build_probe_plan(baseline, "baseline-1")

    assert plan.baseline == baseline
    assert plan.variants == []


# ---------------------------------------------------------------------------
# Requirement 7.5/7.6: batch splitting and baseline-first ordering.
# ---------------------------------------------------------------------------


def test_split_into_batches_keeps_single_batch_at_exactly_max_batch_size() -> None:
    payloads = [{"cells": {}} for _ in range(MAX_BATCH_SIZE)]

    batches = agent.split_into_batches(payloads)

    assert len(batches) == 1
    assert batches[0] == payloads


def test_split_into_batches_splits_ordered_with_baseline_first_when_over_max_batch_size() -> None:
    payloads = [{"cells": {"marker": index}} for index in range(MAX_BATCH_SIZE + 5)]

    batches = agent.split_into_batches(payloads)

    assert len(batches) == 2
    assert len(batches[0]) == MAX_BATCH_SIZE
    assert len(batches[1]) == 5
    # Order preserved across the whole split, and the very first entry of the
    # very first batch is whatever was first in the input (the baseline, by
    # parameter_set_payloads' construction).
    assert batches[0] + batches[1] == payloads
    assert batches[0][0] == payloads[0]


def test_split_into_batches_of_empty_input_returns_no_batches() -> None:
    assert agent.split_into_batches([]) == []


# ---------------------------------------------------------------------------
# Requirement 7.7/7.10: percent change, including the undefined-baseline case.
# ---------------------------------------------------------------------------


def _stub_predict_batch(responses: list[dict[str, Any]]) -> Any:
    """Return a ``predict_batch``-shaped callable that replays ``responses`` in order."""
    calls: list[list[dict[str, Any]]] = []

    def _predict_batch(*, parameter_sets: list[dict[str, Any]], time_step: int | None = None) -> dict[str, Any]:
        calls.append(parameter_sets)
        return responses[len(calls) - 1]

    _predict_batch.calls = calls  # type: ignore[attr-defined]
    return _predict_batch


def _simple_plan(cell_id: str = "gNB_5G") -> Any:
    from smo.aimlfw.common.models import ParameterSet

    baseline = ParameterSet(cells={cell_id: CellParameters.model_validate(_cell_parameters())})
    return agent.build_probe_plan(baseline, "baseline-1")


def _batch_response(
    *, model_name: str = "gnn", model_version: int = 1, target_kpi_values: list[dict[str, float]]
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "model_version": model_version,
        "applied_time_step": 0,
        "predictions": [
            {"index": index, "target_kpi": values, "cell_kpi": {"gNB_5G": values}}
            for index, values in enumerate(target_kpi_values)
        ],
    }


def test_execute_probe_plan_computes_rounded_percent_change_per_target_kpi() -> None:
    plan = _simple_plan()
    variant_count = len(plan.variants)
    # baseline=10.0, first variant=11.0 (+10.0%), remaining variants=9.97 (~-0.3%, exercises rounding).
    values = [{"cell_goodput_mbps": 10.0}, {"cell_goodput_mbps": 11.0}]
    values.extend({"cell_goodput_mbps": 9.97} for _ in range(variant_count - 1))
    predict_batch = _stub_predict_batch([_batch_response(target_kpi_values=values)])

    result = agent.execute_probe_plan(plan, predict_batch=predict_batch)

    assert result.model_name == "gnn"
    assert result.model_version == 1
    assert result.baseline_prediction.target_kpi["cell_goodput_mbps"] == 10.0
    assert result.variant_results[0].percent_change["cell_goodput_mbps"] == pytest.approx(10.0)
    assert result.variant_results[1].percent_change["cell_goodput_mbps"] == pytest.approx(-0.3)


def test_execute_probe_plan_reports_undefined_when_baseline_prediction_is_zero() -> None:
    plan = _simple_plan()
    variant_count = len(plan.variants)
    values = [{"cell_goodput_mbps": 0.0}] + [{"cell_goodput_mbps": 5.0} for _ in range(variant_count)]
    predict_batch = _stub_predict_batch([_batch_response(target_kpi_values=values)])

    result = agent.execute_probe_plan(plan, predict_batch=predict_batch)

    assert result.baseline_prediction.target_kpi["cell_goodput_mbps"] == 0.0
    for variant_result in result.variant_results:
        assert variant_result.percent_change["cell_goodput_mbps"] == "undefined"
        # Raw baseline/variant prediction values are still available.
        assert variant_result.prediction.target_kpi["cell_goodput_mbps"] == 5.0


# ---------------------------------------------------------------------------
# Requirement 7.8: empty_request rejection.
# ---------------------------------------------------------------------------


def test_build_and_execute_probe_plan_rejects_request_with_no_target_cells() -> None:
    empty_request = XAppParameterRequest.model_construct(
        target_cells=[], parameter_overrides={}, xapp_objective=None
    )

    with pytest.raises(agent.ProbeExecutionError) as excinfo:
        agent.build_and_execute_probe_plan(empty_request, feature_store=_FakeFeatureStore([]))
    assert excinfo.value.error_code == "empty_request"


def test_build_and_execute_probe_plan_rejects_request_with_no_parameter_overrides() -> None:
    empty_overrides_request = XAppParameterRequest.model_construct(
        target_cells=["gNB_5G"], parameter_overrides={}, xapp_objective=None
    )

    with pytest.raises(agent.ProbeExecutionError) as excinfo:
        agent.build_and_execute_probe_plan(empty_overrides_request, feature_store=_FakeFeatureStore([]))
    assert excinfo.value.error_code == "empty_request"


# ---------------------------------------------------------------------------
# Requirement 7.11: batch retry-then-succeed and retry-exhausted-then-abort.
# ---------------------------------------------------------------------------


def _error_body(error_code: str = "inference_unavailable") -> dict[str, Any]:
    return {"error_code": error_code, "message": "Inference_Service is unreachable", "details": {}}


def test_execute_probe_plan_retries_a_failing_batch_and_succeeds_within_the_retry_budget() -> None:
    plan = _simple_plan()
    variant_count = len(plan.variants)
    success_values = [{"cell_goodput_mbps": 10.0}] + [
        {"cell_goodput_mbps": 10.0} for _ in range(variant_count)
    ]
    attempts = {"count": 0}

    def flaky_predict_batch(*, parameter_sets: list[dict[str, Any]], time_step: int | None = None) -> dict[str, Any]:
        attempts["count"] += 1
        if attempts["count"] < 3:  # fails twice, succeeds on the 3rd (final allowed) attempt.
            return _error_body()
        return _batch_response(target_kpi_values=success_values)

    result = agent.execute_probe_plan(plan, predict_batch=flaky_predict_batch, max_batch_retries=2)

    assert attempts["count"] == 3
    assert result.model_name == "gnn"


def test_execute_probe_plan_aborts_whole_plan_when_a_batch_exhausts_all_retries() -> None:
    plan = _simple_plan()
    attempts = {"count": 0}

    def always_failing_predict_batch(
        *, parameter_sets: list[dict[str, Any]], time_step: int | None = None
    ) -> dict[str, Any]:
        attempts["count"] += 1
        return _error_body("inference_unavailable")

    with pytest.raises(agent.ProbeExecutionError) as excinfo:
        agent.execute_probe_plan(plan, predict_batch=always_failing_predict_batch, max_batch_retries=2)

    # 1 initial attempt + 2 retries = 3 total attempts, then abort with no partial result.
    assert attempts["count"] == 3
    assert excinfo.value.error_code == "probe_batch_failed"
    assert excinfo.value.details["batch_index"] == 0
    assert excinfo.value.details["attempts"] == 3
    assert excinfo.value.details["underlying_error_code"] == "inference_unavailable"


def test_execute_probe_plan_does_not_retry_a_batch_after_the_first_batch_already_succeeded() -> None:
    """A later batch's failure must not resend earlier, already-succeeded batches."""
    from smo.aimlfw.common.models import ParameterSet

    # 65 Parameter_Sets (baseline + enough variants) forces a 2-batch split.
    baseline = ParameterSet(
        cells={f"cell_{index}": CellParameters.model_validate(_cell_parameters()) for index in range(13)}
    )
    plan = agent.build_probe_plan(baseline, "baseline-1")
    assert len(plan.variants) + 1 > MAX_BATCH_SIZE, "fixture must exercise a real multi-batch split"

    batch_sizes_seen: list[int] = []

    def first_batch_succeeds_second_batch_fails(
        *, parameter_sets: list[dict[str, Any]], time_step: int | None = None
    ) -> dict[str, Any]:
        batch_sizes_seen.append(len(parameter_sets))
        if len(batch_sizes_seen) == 1:
            values = [{"cell_goodput_mbps": 10.0} for _ in parameter_sets]
            return _batch_response(target_kpi_values=values)
        return _error_body("inference_unavailable")

    with pytest.raises(agent.ProbeExecutionError) as excinfo:
        agent.execute_probe_plan(
            plan, predict_batch=first_batch_succeeds_second_batch_fails, max_batch_retries=1
        )

    assert excinfo.value.details["batch_index"] == 1
    # 1 successful first-batch call, then exactly 2 attempts (1 + 1 retry) at
    # the second (failing) batch before the plan aborts -- the first batch is
    # never resent once it has already succeeded.
    assert len(batch_sizes_seen) == 3
    assert batch_sizes_seen[0] == MAX_BATCH_SIZE
    assert batch_sizes_seen[1] == batch_sizes_seen[2]  # both failed attempts hit the same (second) batch.


# ---------------------------------------------------------------------------
# Requirement 7.12: model-version mismatch across batches.
# ---------------------------------------------------------------------------


def test_execute_probe_plan_raises_model_version_mismatch_across_batches() -> None:
    from smo.aimlfw.common.models import ParameterSet

    baseline = ParameterSet(
        cells={f"cell_{index}": CellParameters.model_validate(_cell_parameters()) for index in range(13)}
    )
    plan = agent.build_probe_plan(baseline, "baseline-1")
    assert len(plan.variants) + 1 > MAX_BATCH_SIZE, "fixture must exercise a real multi-batch split"

    call_count = {"value": 0}

    def mismatched_version_predict_batch(
        *, parameter_sets: list[dict[str, Any]], time_step: int | None = None
    ) -> dict[str, Any]:
        call_count["value"] += 1
        version = 1 if call_count["value"] == 1 else 2
        values = [{"cell_goodput_mbps": 10.0} for _ in parameter_sets]
        return _batch_response(model_version=version, target_kpi_values=values)

    with pytest.raises(agent.ProbeExecutionError) as excinfo:
        agent.execute_probe_plan(plan, predict_batch=mismatched_version_predict_batch)

    assert excinfo.value.error_code == "model_version_mismatch"
    assert excinfo.value.details["expected_model_version"] == 1
    assert excinfo.value.details["actual_model_version"] == 2


def test_execute_probe_plan_accepts_matching_model_version_across_batches() -> None:
    from smo.aimlfw.common.models import ParameterSet

    baseline = ParameterSet(
        cells={f"cell_{index}": CellParameters.model_validate(_cell_parameters()) for index in range(13)}
    )
    plan = agent.build_probe_plan(baseline, "baseline-1")
    assert len(plan.variants) + 1 > MAX_BATCH_SIZE, "fixture must exercise a real multi-batch split"

    def consistent_predict_batch(
        *, parameter_sets: list[dict[str, Any]], time_step: int | None = None
    ) -> dict[str, Any]:
        values = [{"cell_goodput_mbps": 10.0} for _ in parameter_sets]
        return _batch_response(model_version=1, target_kpi_values=values)

    result = agent.execute_probe_plan(plan, predict_batch=consistent_predict_batch)

    assert result.model_version == 1
    assert len(result.variant_results) == len(plan.variants)


# ---------------------------------------------------------------------------
# End-to-end orchestration seam.
# ---------------------------------------------------------------------------


def test_build_and_execute_probe_plan_end_to_end_with_fake_feature_store_and_stub_predict_batch() -> None:
    store = _FakeFeatureStore(
        [_feature_record(cell_id="gNB_5G", time_step=0, control_parameters=_cell_parameters())]
    )
    xapp_request = XAppParameterRequest(
        target_cells=["gNB_5G"],
        parameter_overrides={"gNB_5G": PartialCellParameters(tx_power_dbm=41.0)},
    )

    def predict_batch(*, parameter_sets: list[dict[str, Any]], time_step: int | None = None) -> dict[str, Any]:
        values = [{"cell_goodput_mbps": 10.0 + index} for index in range(len(parameter_sets))]
        return _batch_response(target_kpi_values=values)

    result = agent.build_and_execute_probe_plan(
        xapp_request, feature_store=store, predict_batch=predict_batch
    )

    assert result.plan.baseline.cells["gNB_5G"].tx_power_dbm == 41.0
    assert len(result.variant_results) == len(result.plan.variants)
