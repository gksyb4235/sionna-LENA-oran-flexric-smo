"""Property 44 test module (own module to avoid collisions with parallel Property tasks).

Task 7.7: Property 44 속성 테스트 작성: 변화율 계산.

Targets ``agent.execute_probe_plan`` (and the ``agent._percent_change`` helper
it calls), task 7.2's Requirement 7.7/7.10 implementation.

Design tag (design.md, "KPI_Advisor_Agent -- 배치 프로빙 (Requirement 7)"):

    #### Property 44: 변화율 계산
    *For any* Baseline과 Probe_Variant의 Target_KPI 예측값 쌍, 변화율은
    `(variant - baseline) / |baseline| * 100`을 소수점 첫째 자리로 반올림한
    값과 일치하되, baseline 예측값의 절대값이 0인 경우 변화율은 `undefined`로
    표기되고 원시 예측값이 그대로 반환된다.
    **Validates: Requirements 7.7, 7.10**

This module upgrades the fixed-example coverage already present in
``test_agent_baseline_and_probe_plan.py`` (task 7.2:
``test_execute_probe_plan_computes_rounded_percent_change_per_target_kpi``,
``test_execute_probe_plan_reports_undefined_when_baseline_prediction_is_zero``)
into genuine Hypothesis property tests sweeping many generated
(baseline_value, variant_value) pairs across varied Target_KPI names, rather
than the handful of fixed examples already present there.

Testing seam choice: this module tests the bare ``agent._percent_change``
helper directly for points 1 and 2 below (it is just a module-level pure
function, not enforced-private by Python, and reaching it avoids
constructing a full ``ProbePlan``/stub ``predict_batch`` for every one of the
many Hypothesis examples). Point 3 (raw baseline/variant prediction values
surviving alongside an ``"undefined"`` percent change) can only be observed
at the ``execute_probe_plan`` level, since ``_percent_change`` itself never
returns raw values -- so that sub-requirement is covered by one supplementary
test exercising ``execute_probe_plan`` with a stub ``predict_batch``,
following ``test_agent_baseline_and_probe_plan.py``'s existing stubbing
convention.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

AGENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(AGENT_DIR))

from smo.aimlfw.common.constants import TARGET_KPIS  # noqa: E402
from smo.aimlfw.common.models import CellParameters, ParameterSet  # noqa: E402

import agent  # noqa: E402

PROPERTY_TEST_SETTINGS = settings(max_examples=100, deadline=None)

# Baseline/variant values with occasional exact 0.0 baselines mixed in --
# st.floats alone would essentially never draw exactly 0.0, so it is added
# explicitly via st.just(0.0) (per the task's point 2 requirement).
_KPI_VALUE = st.floats(
    min_value=-1_000_000.0,
    max_value=1_000_000.0,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)
_NONZERO_BASELINE_VALUE = st.one_of(
    st.floats(min_value=-1_000_000.0, max_value=-1e-9, allow_nan=False, allow_infinity=False, width=64),
    st.floats(min_value=1e-9, max_value=1_000_000.0, allow_nan=False, allow_infinity=False, width=64),
)
_KPI_NAME = st.sampled_from(TARGET_KPIS)


def _oracle_percent_change(baseline_value: float, variant_value: float) -> float:
    """Independently compute Requirement 7.7's exact formula, rounded the same way as the code under test."""
    return round((variant_value - baseline_value) / abs(baseline_value) * 100, 1)


# ---------------------------------------------------------------------------
# Point 1: baseline_value != 0 -> percent change matches the oracle formula.
# ---------------------------------------------------------------------------


# **Property 44: 변화율 계산**
# **Validates: Requirements 7.7, 7.10**
@PROPERTY_TEST_SETTINGS
@given(
    target_kpi=_KPI_NAME,
    baseline_value=_NONZERO_BASELINE_VALUE,
    variant_value=_KPI_VALUE,
)
def test_percent_change_matches_oracle_formula_for_nonzero_baseline(
    target_kpi: str, baseline_value: float, variant_value: float
) -> None:
    changes = agent._percent_change({target_kpi: baseline_value}, {target_kpi: variant_value})

    expected = _oracle_percent_change(baseline_value, variant_value)
    assert changes[target_kpi] == expected


# **Property 44: 변화율 계산**
# **Validates: Requirements 7.7, 7.10**
@PROPERTY_TEST_SETTINGS
@given(
    kpi_names=st.lists(_KPI_NAME, min_size=1, max_size=len(TARGET_KPIS), unique=True),
    values=st.data(),
)
def test_percent_change_matches_oracle_formula_across_multiple_target_kpis_at_once(
    kpi_names: list[str], values: st.DataObject
) -> None:
    baseline_kpi = {name: values.draw(_NONZERO_BASELINE_VALUE, label=f"baseline-{name}") for name in kpi_names}
    variant_kpi = {name: values.draw(_KPI_VALUE, label=f"variant-{name}") for name in kpi_names}

    changes = agent._percent_change(baseline_kpi, variant_kpi)

    for name in kpi_names:
        assert changes[name] == _oracle_percent_change(baseline_kpi[name], variant_kpi[name])


# ---------------------------------------------------------------------------
# Point 2: baseline_value == 0 (exact) -> "undefined", regardless of variant_value.
# ---------------------------------------------------------------------------


# **Property 44: 변화율 계산**
# **Validates: Requirements 7.7, 7.10**
@PROPERTY_TEST_SETTINGS
@given(target_kpi=_KPI_NAME, variant_value=st.one_of(st.just(0.0), _KPI_VALUE))
def test_percent_change_is_the_string_undefined_for_exact_zero_baseline(
    target_kpi: str, variant_value: float
) -> None:
    changes = agent._percent_change({target_kpi: 0.0}, {target_kpi: variant_value})

    result = changes[target_kpi]
    assert result == "undefined"
    assert isinstance(result, str)  # not a float: must not be confused with a numeric 0.0 percent change.


def test_percent_change_is_undefined_for_several_explicit_zero_baseline_variant_pairs() -> None:
    """Explicit sweep of variant_value against a zero baseline: zero, positive, and negative."""
    for variant_value in (0.0, 5.0, -5.0, 1_000_000.0, -1_000_000.0):
        changes = agent._percent_change(
            {"cell_goodput_mbps": 0.0}, {"cell_goodput_mbps": variant_value}
        )
        assert changes["cell_goodput_mbps"] == "undefined"


# ---------------------------------------------------------------------------
# Point 3: raw baseline/variant prediction values survive at the
# execute_probe_plan level even when percent_change is "undefined".
# ---------------------------------------------------------------------------


_BASE_CONTROL_PARAMETERS: dict[str, float | int] = {
    "tx_power_dbm": 40.0,
    "ret_tilt_deg": 5.0,
    "cio_bias_db": 0.0,
    "hysteresis_db": 2.0,
    "ttt_ms": 160,
}


def _batch_response(*, target_kpi_values: list[dict[str, float]]) -> dict[str, Any]:
    return {
        "model_name": "gnn",
        "model_version": 1,
        "applied_time_step": 0,
        "predictions": [
            {"index": index, "target_kpi": values, "cell_kpi": {"gNB_5G": values}}
            for index, values in enumerate(target_kpi_values)
        ],
    }


# **Property 44: 변화율 계산**
# **Validates: Requirements 7.7, 7.10**
@PROPERTY_TEST_SETTINGS
@given(variant_value=_KPI_VALUE)
def test_execute_probe_plan_preserves_raw_baseline_and_variant_values_when_percent_change_is_undefined(
    variant_value: float,
) -> None:
    baseline = ParameterSet(cells={"gNB_5G": CellParameters.model_validate(_BASE_CONTROL_PARAMETERS)})
    plan = agent.build_probe_plan(baseline, "baseline-1")
    variant_count = len(plan.variants)

    values = [{"cell_goodput_mbps": 0.0}] + [
        {"cell_goodput_mbps": variant_value} for _ in range(variant_count)
    ]

    def predict_batch(*, parameter_sets: list[dict[str, Any]], time_step: int | None = None) -> dict[str, Any]:
        return _batch_response(target_kpi_values=values)

    result = agent.execute_probe_plan(plan, predict_batch=predict_batch)

    # The raw baseline prediction value (0.0) is preserved, not just the "undefined" marker.
    assert result.baseline_prediction.target_kpi["cell_goodput_mbps"] == 0.0
    for variant_result in result.variant_results:
        assert variant_result.percent_change["cell_goodput_mbps"] == "undefined"
        # The raw variant prediction value is preserved too, regardless of variant_value's sign/magnitude.
        assert variant_result.prediction.target_kpi["cell_goodput_mbps"] == variant_value
