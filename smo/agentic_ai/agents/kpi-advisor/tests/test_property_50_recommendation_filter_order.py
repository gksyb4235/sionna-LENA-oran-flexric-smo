"""Property 50: recommendations are eligible, deterministic, and capped at five."""

from __future__ import annotations

import sys
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

TEST_DIR = Path(__file__).resolve().parent
AGENT_DIR = TEST_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(REPO_ROOT), str(AGENT_DIR), str(TEST_DIR)]

import test_agent_threshold_judgment_and_recommendations as cases  # noqa: E402


# **Property 50: 추천 후보 필터링과 결정론적 정렬**
# **Validates: Requirements 8.5~8.8**
@settings(max_examples=100, deadline=None)
@given(improvements=st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=10, unique=True))
def test_recommendations_are_sorted_by_improvement_then_capped(improvements: list[int]) -> None:
    baseline = cases._simple_baseline()
    base_variant = cases._variant(
        baseline,
        cell_id="gNB_5G",
        parameter_name="tx_power_dbm",
        direction="+1",
        variant_id="base",
    )
    results = []
    for index, improvement in enumerate(improvements):
        variant = base_variant.model_copy(update={"variant_id": f"variant-{index:02d}"})
        results.append(cases._variant_result(variant, cell_goodput_mbps=100.0, improvement_percent=float(improvement)))
    outcome = cases._recommend(
        results,
        cases._default_kpi_threshold_config(),
        cases._default_objective_primary_kpi_config(),
    )
    actual = [(item.improvement_percent, item.variant_id) for item in outcome.recommendations]
    expected = sorted(
        [(float(value), f"variant-{index:02d}") for index, value in enumerate(improvements)],
        key=lambda item: (-item[0], item[1]),
    )[:5]
    assert actual == expected
