"""Property 43 test module (own module to avoid collisions with parallel Property tasks).

Task 7.6: 배치 분할 순서 보존.

Targets ``smo/agentic_ai/agents/kpi-advisor/agent.py``'s (task 7.2)
``split_into_batches``/``parameter_set_payloads`` functions directly, as
specified in requirements.md:

Requirement 7.5:
    Probe_Plan 의 Parameter_Set 개수가 64 이하이면, 모든 Parameter_Set 을
    Probe_Plan 기록 순서대로 하나의 Batch_Prediction_Request 에 담아 전달한다.

Requirement 7.6:
    64 를 초과하면, 순서를 유지한 채 각 배치가 64 이하가 되도록 분할하고,
    Baseline_Parameter_Set 을 첫 배치의 첫 Parameter_Set 으로 배치하여 모든
    배치를 순차 전달한다.

Design tag (design.md, "KPI_Advisor_Agent -- 배치 프로빙 (Requirement 7)"):

    #### Property 43: 배치 분할 순서 보존
    *For any* 크기가 64를 초과하는 Probe_Plan, 분할된 배치들을 원래 순서대로
    재결합하면 원본 Probe_Plan의 Parameter_Set 순서와 일치하고, 각 배치의
    크기는 64 이하이며, 첫 배치의 첫 원소는 Baseline_Parameter_Set이다.
    **Validates: Requirements 7.5, 7.6**

This module upgrades the fixed-example coverage already present in
``test_agent_baseline_and_probe_plan.py`` (task 7.2:
``test_split_into_batches_keeps_single_batch_at_exactly_max_batch_size``,
``test_split_into_batches_splits_ordered_with_baseline_first_when_over_max_batch_size``,
``test_split_into_batches_of_empty_input_returns_no_batches``) into genuine
Hypothesis property tests sweeping many generated input sizes/shapes -- both
at the default ``MAX_BATCH_SIZE=64`` and across varied non-default
``max_batch_size`` values, to exercise the general splitting mechanism
rather than only the hardcoded 64 case. ``split_into_batches`` operates on
generic dicts and never inspects their contents, so generated payloads are
plain ``{"marker": index}`` dicts rather than real Parameter_Set payloads --
this is sufficient to verify identity/order by inspecting the marker.

A final test exercises the real integration point (``build_probe_plan`` ->
``parameter_set_payloads`` -> ``split_into_batches``) to prove the
baseline-first guarantee end-to-end against a real ``ProbePlan``'s baseline,
not just a generic dict.

Follows ``test_agent_baseline_and_probe_plan.py``'s manual ``sys.path``
setup convention since ``kpi-advisor`` is not installed as a package during
tests, and follows ``smo/tests/property``'s ``PROPERTY_TEST_SETTINGS``/
design-tag convention for Hypothesis-based property modules.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

AGENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(AGENT_DIR))

from smo.aimlfw.common.models import CellParameters, ParameterSet  # noqa: E402
from smo.aimlfw.inference_service.schemas import MAX_BATCH_SIZE  # noqa: E402

import agent  # noqa: E402

PROPERTY_TEST_SETTINGS = settings(max_examples=50, deadline=None)

_BASE_CONTROL_PARAMETERS: dict[str, float | int] = {
    "tx_power_dbm": 40.0,
    "ret_tilt_deg": 5.0,
    "cio_bias_db": 0.0,
    "hysteresis_db": 2.0,
    "ttt_ms": 160,
}


def _markers(size: int) -> list[dict[str, Any]]:
    return [{"marker": index} for index in range(size)]


def _assert_order_preserving_split(
    payloads: list[dict[str, Any]], batches: list[list[dict[str, Any]]], *, max_batch_size: int
) -> None:
    # (a) every batch's length is <= max_batch_size.
    for batch in batches:
        assert len(batch) <= max_batch_size

    # (b) concatenating all batches in order reproduces the original input exactly.
    assert sum(batches, []) == payloads

    # (c) batch count is ceil(len(payloads) / max_batch_size), 0 when empty.
    if payloads:
        assert len(batches) == math.ceil(len(payloads) / max_batch_size)
    else:
        assert batches == []

    # (d) the very first element of the very first batch equals payloads[0]
    # (the generic, baseline-agnostic half of the "baseline first" invariant).
    if payloads:
        assert batches[0][0] == payloads[0]


# **Property 43: 배치 분할 순서 보존**
# **Validates: Requirements 7.5, 7.6**
@PROPERTY_TEST_SETTINGS
@given(size=st.integers(min_value=0, max_value=300))
def test_split_into_batches_preserves_order_and_bounds_at_default_max_batch_size(size: int) -> None:
    payloads = _markers(size)

    batches = agent.split_into_batches(payloads)

    _assert_order_preserving_split(payloads, batches, max_batch_size=MAX_BATCH_SIZE)


# **Property 43: 배치 분할 순서 보존**
# **Validates: Requirements 7.5, 7.6**
@PROPERTY_TEST_SETTINGS
@given(
    size=st.integers(min_value=0, max_value=300),
    max_batch_size=st.integers(min_value=1, max_value=200),
)
def test_split_into_batches_preserves_order_and_bounds_across_varied_max_batch_size(
    size: int, max_batch_size: int
) -> None:
    payloads = _markers(size)

    batches = agent.split_into_batches(payloads, max_batch_size=max_batch_size)

    _assert_order_preserving_split(payloads, batches, max_batch_size=max_batch_size)


# **Property 43: 배치 분할 순서 보존**
# **Validates: Requirements 7.5, 7.6**
def test_split_into_batches_places_real_probe_plan_baseline_first_end_to_end() -> None:
    """The real integration point: build_probe_plan -> parameter_set_payloads -> split_into_batches.

    13 target cells x 5 Control_Parameters x 2 directions = 130 Probe_Variants
    (every direction here is strictly interior to its allowed range/allowed-
    value set), plus the Baseline_Parameter_Set itself, totals 131
    Parameter_Sets -- comfortably over MAX_BATCH_SIZE (64), forcing a real
    multi-batch split (reusing the ``cell_{index}`` multi-cell construction
    pattern from test_agent_baseline_and_probe_plan.py's multi-batch tests).
    """
    baseline = ParameterSet(
        cells={
            f"cell_{index}": CellParameters.model_validate(_BASE_CONTROL_PARAMETERS) for index in range(13)
        }
    )
    plan = agent.build_probe_plan(baseline, "baseline-1")
    assert len(plan.variants) + 1 > MAX_BATCH_SIZE, "fixture must exercise a real multi-batch split"

    payloads = agent.parameter_set_payloads(plan)
    batches = agent.split_into_batches(payloads)

    assert len(batches) > 1
    first_payload_cells = batches[0][0]["cells"]
    assert first_payload_cells == plan.baseline.model_dump(mode="json")["cells"]
