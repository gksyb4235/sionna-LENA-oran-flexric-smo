"""Property 42 test module (own module to avoid collisions with parallel Property tasks).

Task 7.5: Property 42 속성 테스트 작성: Probe_Plan variant 불변식.

Targets ``agent.build_probe_plan`` (task 7.2, Requirement 7.2/7.3/7.4).

Design tag (design.md, "KPI_Advisor_Agent -- 배치 프로빙 (Requirement 7)"):

    #### Property 42: Probe_Plan variant 불변식
    *For any* Baseline_Parameter_Set과 대상 셀 집합, 생성된 모든 Probe_Variant는
    Baseline과 정확히 하나의 (cell_id, Control_Parameter) 값만 다르며, 그 차이는
    해당 파라미터의 스텝 크기의 정확히 +1배 또는 -1배이고, 허용 범위/허용 값
    집합을 벗어나는 방향의 variant는 생성되지 않는다(모든 방향이 범위를 벗어나면
    variant 개수는 0).
    **Validates: Requirements 7.2, 7.3, 7.4**

``ProbePlan``'s own Pydantic validator (smo.aimlfw.common.models) already
enforces unique ``variant_id`` values, that every variant references
``baseline_id``, that every variant's ``cell_id`` is a target cell, and that
every variant's ``parameter_set`` covers exactly the target cells -- simply
constructing a ``ProbePlan`` at all (which ``build_probe_plan`` always
returns) already proves those invariants. This module instead focuses on the
*content*-level invariants the schema's validator does not check: that each
(cell_id, Control_Parameter) pair gets exactly the +1/-1 variants
``probing.step_value`` says are in range (no more, no less), that each
variant differs from baseline in exactly one (cell_id, Control_Parameter)
value, and that the total variant count matches the in-range-direction sum.

Oracle grounding: every expectation below is computed independently via
``probing.step_value`` (the same function ``build_probe_plan`` itself calls)
rather than via hardcoded per-parameter constants, so the test is driven by
the single-sourced stepping rule instead of duplicating it.

This module deliberately goes beyond ``test_agent_baseline_and_probe_plan.py``
(task 7.2)'s fixed/boundary examples by sweeping many generated baseline
``ParameterSet``s -- both a general sweep via ``valid_parameter_sets()`` and a
boundary-leaning sweep that deliberately pins some (cell, parameter) values to
range/allowed-value boundaries so some directions are genuinely out of range
across many combinations, not just by chance.

Finding on Requirement 7.4 (documented, not asserted as a Hypothesis
property): given this domain's actual ``CONTROL_PARAMETER_RANGES``/
``TTT_ALLOWED_MS`` (every continuous parameter's range width is at least 10x
its step size; ``TTT_ALLOWED_MS`` has 16 entries), no single (cell,
Control_Parameter) pair can ever have *both* directions out of range
simultaneously -- only one boundary can be "hugged" at a time. Consequently
Requirement 7.4's "every direction out of range for every parameter" case is
structurally unreachable through real ``step_value`` outputs for *any* real
Baseline_Parameter_Set; ``test_agent_baseline_and_probe_plan.py`` already
covers that requirement's zero-variant behavior via a monkeypatched
``step_value`` (the only way to reach it). ``test_step_value_never_reports_
both_directions_out_of_range_at_either_boundary`` below verifies this domain
fact directly instead of attempting to (impossibly) construct a real
zero-variant Probe_Plan.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

AGENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(AGENT_DIR))

from smo.aimlfw.common.constants import CONTROL_PARAMETER_NAMES, TTT_ALLOWED_MS  # noqa: E402
from smo.aimlfw.common.models import CellParameters, ParameterSet  # noqa: E402
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS, valid_parameter_sets  # noqa: E402

import agent  # noqa: E402
from probing import STEP_DIRECTIONS, step_value  # noqa: E402

# One (min, max) pair per Control_Parameter, matching the boundary values the
# task calls out explicitly: tx_power_dbm 30/46, ret_tilt_deg 0/15,
# cio_bias_db -6.0/6.0, hysteresis_db 0.0/10.0, ttt_ms first/last allowed value.
_BOUNDARY_VALUES: dict[str, tuple[float | int, float | int]] = {
    "tx_power_dbm": (30.0, 46.0),
    "ret_tilt_deg": (0.0, 15.0),
    "cio_bias_db": (-6.0, 6.0),
    "hysteresis_db": (0.0, 10.0),
    "ttt_ms": (TTT_ALLOWED_MS[0], TTT_ALLOWED_MS[-1]),
}


@st.composite
def _boundary_leaning_parameter_sets(draw):
    """Generate ``valid_parameter_sets()`` with some (cell, parameter) values pinned to boundaries.

    For every generated cell, each Control_Parameter is independently either
    left as-is (an interior, generally-both-directions-valid value from
    ``valid_parameter_sets()``) or overridden to one of that parameter's two
    range/allowed-value boundaries. This deliberately forces some directions
    out of range across the Hypothesis sweep, rather than leaving it to
    chance, while still producing plenty of interior values too.
    """
    baseline = draw(valid_parameter_sets(min_cells=1, max_cells=3))
    updated_cells = dict(baseline.cells)
    for cell_id, cell_parameters in baseline.cells.items():
        overrides: dict[str, float | int] = {}
        for parameter_name in CONTROL_PARAMETER_NAMES:
            if draw(st.booleans()):
                low, high = _BOUNDARY_VALUES[parameter_name]
                overrides[parameter_name] = draw(st.sampled_from((low, high)))
        if overrides:
            updated_cells[cell_id] = cell_parameters.model_copy(update=overrides)
    return ParameterSet(cells=updated_cells)


def _expected_variants(baseline: ParameterSet) -> dict[tuple[str, str, str], float | int]:
    """Independently compute (via ``step_value``) which variants must exist, and their target values."""
    expected: dict[tuple[str, str, str], float | int] = {}
    for cell_id, cell_parameters in baseline.cells.items():
        for parameter_name in CONTROL_PARAMETER_NAMES:
            current_value = getattr(cell_parameters, parameter_name)
            for direction in STEP_DIRECTIONS:
                new_value = step_value(parameter_name, current_value, direction)
                if new_value is not None:
                    expected[(cell_id, parameter_name, direction)] = new_value
    return expected


def _assert_probe_plan_matches_oracle(baseline: ParameterSet, plan: agent.ProbePlan) -> None:
    """Assert ``plan.variants`` matches the ``step_value``-derived oracle exactly (points 1-3)."""
    expected = _expected_variants(baseline)

    # Point 1: a variant with a given (cell_id, parameter_name, direction) key
    # exists if and only if step_value(...) returned non-None for it.
    actual_keys = {(variant.cell_id, variant.parameter_name, variant.direction) for variant in plan.variants}
    assert actual_keys == set(expected)

    # Point 3: total variant count equals the sum, over every (cell,
    # parameter), of how many of {+1, -1} are in range.
    assert len(plan.variants) == len(expected)

    for variant in plan.variants:
        # Point 2: exactly one (cell_id, Control_Parameter) differs from baseline.
        differing = [
            (cell_id, parameter_name)
            for cell_id, cell_parameters in baseline.cells.items()
            for parameter_name in CONTROL_PARAMETER_NAMES
            if getattr(cell_parameters, parameter_name)
            != getattr(variant.parameter_set.cells[cell_id], parameter_name)
        ]
        assert differing == [(variant.cell_id, variant.parameter_name)]

        # The stepped value itself matches the independently computed oracle value.
        key = (variant.cell_id, variant.parameter_name, variant.direction)
        expected_value = expected[key]
        actual_value = getattr(variant.parameter_set.cells[variant.cell_id], variant.parameter_name)
        assert actual_value == expected_value

        # Untouched cells (when there are multiple target cells) are byte-for-byte unchanged.
        for other_cell_id, other_cell_parameters in baseline.cells.items():
            if other_cell_id != variant.cell_id:
                assert variant.parameter_set.cells[other_cell_id] == other_cell_parameters


# **Property 42: Probe_Plan variant 불변식**
# **Validates: Requirements 7.2, 7.3, 7.4**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(baseline=valid_parameter_sets(min_cells=1, max_cells=3))
def test_build_probe_plan_matches_step_value_oracle_for_generated_baselines(baseline: ParameterSet) -> None:
    plan = agent.build_probe_plan(baseline, "baseline-1")
    _assert_probe_plan_matches_oracle(baseline, plan)


# **Property 42: Probe_Plan variant 불변식**
# **Validates: Requirements 7.2, 7.3, 7.4**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(baseline=_boundary_leaning_parameter_sets())
def test_build_probe_plan_matches_step_value_oracle_for_boundary_leaning_baselines(
    baseline: ParameterSet,
) -> None:
    plan = agent.build_probe_plan(baseline, "baseline-1")
    _assert_probe_plan_matches_oracle(baseline, plan)


# ---------------------------------------------------------------------------
# Requirement 7.4 structural finding: is "every direction out of range for
# every parameter, simultaneously" reachable through real step_value outputs?
# ---------------------------------------------------------------------------


def test_step_value_never_reports_both_directions_out_of_range_at_either_boundary() -> None:
    """No single (cell, Control_Parameter) pair can have zero valid directions.

    This domain's ranges (tx_power_dbm 30-46 step 1, ret_tilt_deg 0-15 step 1,
    cio_bias_db -6..6 step 0.5, hysteresis_db 0..10 step 0.5) are all far
    wider than twice their step size, and ``TTT_ALLOWED_MS`` has 16 entries --
    so at either boundary, exactly one direction is out of range and the
    other always remains valid. This directly verifies, via the same
    ``step_value`` oracle ``build_probe_plan`` uses, that Requirement 7.4's
    "every direction out of range" case can never occur for a single
    (cell, parameter) pair with real domain values, and therefore never for
    *all five* parameters of a cell simultaneously either. A genuine
    zero-variant Probe_Plan is thus only reachable in this codebase via the
    monkeypatched ``step_value`` already exercised by
    ``test_agent_baseline_and_probe_plan.py`` (task 7.2), not through any real
    Baseline_Parameter_Set -- this test documents that finding structurally.
    """
    for parameter_name in CONTROL_PARAMETER_NAMES:
        low, high = _BOUNDARY_VALUES[parameter_name]

        # At the lower boundary: -1 must be out of range, +1 must remain valid.
        assert step_value(parameter_name, low, "-1") is None
        assert step_value(parameter_name, low, "+1") is not None

        # At the upper boundary: +1 must be out of range, -1 must remain valid.
        assert step_value(parameter_name, high, "+1") is None
        assert step_value(parameter_name, high, "-1") is not None


def test_build_probe_plan_drops_exactly_one_direction_when_one_parameter_sits_at_a_boundary() -> None:
    """Changing one parameter to a boundary value removes exactly that one expected direction.

    A fixed, fully-interior baseline has both directions available for every
    Control_Parameter (10 variants for 1 cell). Moving ``tx_power_dbm`` to its
    upper boundary (46) removes only the ``+1`` variant for that one
    parameter -- every other (cell, parameter, direction) combination is
    unaffected. This is the achievable, real (non-monkeypatched) counterpart
    to Requirement 7.4's "direction removed" behavior, exercised at the
    single-(cell, parameter) granularity that is actually reachable in this
    domain (see the structural finding above for why "zero directions for
    every parameter simultaneously" is not).
    """
    interior_values = {
        "tx_power_dbm": 40.0,
        "ret_tilt_deg": 5.0,
        "cio_bias_db": 0.0,
        "hysteresis_db": 2.0,
        "ttt_ms": 160,
    }
    interior_baseline = ParameterSet(cells={"gNB_5G": CellParameters.model_validate(interior_values)})
    boundary_baseline = ParameterSet(
        cells={"gNB_5G": CellParameters.model_validate({**interior_values, "tx_power_dbm": 46.0})}
    )

    interior_plan = agent.build_probe_plan(interior_baseline, "baseline-1")
    boundary_plan = agent.build_probe_plan(boundary_baseline, "baseline-1")

    _assert_probe_plan_matches_oracle(interior_baseline, interior_plan)
    _assert_probe_plan_matches_oracle(boundary_baseline, boundary_plan)

    assert len(interior_plan.variants) == 2 * len(CONTROL_PARAMETER_NAMES)
    assert len(boundary_plan.variants) == len(interior_plan.variants) - 1

    interior_keys = {(v.cell_id, v.parameter_name, v.direction) for v in interior_plan.variants}
    boundary_keys = {(v.cell_id, v.parameter_name, v.direction) for v in boundary_plan.variants}
    assert interior_keys - boundary_keys == {("gNB_5G", "tx_power_dbm", "+1")}
