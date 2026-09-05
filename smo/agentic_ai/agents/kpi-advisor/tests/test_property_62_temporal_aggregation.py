"""Property 62: five-step KPI aggregation follows configured sum/mean rules."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

TEST_DIR = Path(__file__).resolve().parent
AGENT_DIR = TEST_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(REPO_ROOT), str(AGENT_DIR), str(TEST_DIR)]

import test_temporal_scheduler_smoke as cases  # noqa: E402


# **Property 62: 5구간 누적값 집계 규칙**
# **Validates: Requirement 10.5**
@settings(max_examples=100, deadline=None)
@given(base=st.integers(min_value=1, max_value=100), delta=st.integers(min_value=-5, max_value=5))
def test_sum_and_mean_aggregations_match_oracle(base: int, delta: int) -> None:
    def kpis(_cells: dict[str, object], step: int | None) -> dict[str, float]:
        value = float(base + (step or 0) * delta)
        return cases._default_kpis(cell_goodput_mbps=value, sinr_p50_db=value)

    plan = cases._build(
        objectives=["throughput_maximization"],
        candidates_by_objective={"throughput_maximization": [cases._parameter_set()]},
        predict_batch=cases._make_predict_batch(kpis),
    )
    values = [base + step * delta for step in range(5)]
    assert plan.aggregate_kpis["cell_goodput_mbps"] == pytest.approx(sum(values))
    assert plan.aggregate_kpis["sinr_p50_db"] == pytest.approx(sum(values) / 5)
