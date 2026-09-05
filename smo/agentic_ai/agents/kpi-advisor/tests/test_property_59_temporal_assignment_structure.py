"""Property 59: temporal planning selects the optimum and always returns five complete steps."""

from __future__ import annotations

import sys
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

TEST_DIR = Path(__file__).resolve().parent
AGENT_DIR = TEST_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(REPO_ROOT), str(AGENT_DIR), str(TEST_DIR)]

import test_temporal_scheduler_smoke as cases  # noqa: E402


# **Property 59: Time_Step별 최적 배정과 구조 완전성**
# **Validates: Requirements 10.1, 10.4, 10.6, 10.10**
@settings(max_examples=100, deadline=None)
@given(powers=st.lists(st.integers(min_value=30, max_value=46), min_size=1, max_size=12, unique=True))
def test_best_candidate_is_selected_for_all_five_steps(powers: list[int]) -> None:
    candidates = [cases._parameter_set(tx_power_dbm=float(power)) for power in powers]
    predictor = cases._make_predict_batch(
        lambda cells, _step: cases._default_kpis(cell_goodput_mbps=cells["gNB_5G"]["tx_power_dbm"])
    )
    plan = cases._build(
        objectives=["throughput_maximization"],
        candidates_by_objective={"throughput_maximization": candidates},
        predict_batch=predictor,
    )
    assert [item.time_step for item in plan.assignments] == [0, 1, 2, 3, 4]
    assert all(item.parameter_set.cells["gNB_5G"].tx_power_dbm == max(powers) for item in plan.assignments)
    assert all("cell_goodput_mbps" in item.predicted_kpis for item in plan.assignments)
