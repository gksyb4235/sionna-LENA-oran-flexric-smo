"""Property 63: transition_time_steps exactly identify objective changes."""

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


# **Property 63: 전환 목록 계산**
# **Validates: Requirement 10.7**
@settings(max_examples=100, deadline=None)
@given(switch_step=st.integers(min_value=1, max_value=4))
def test_transition_list_contains_exact_switch_step(switch_step: int) -> None:
    throughput = cases._parameter_set(ret_tilt_deg=1.0)
    energy = cases._parameter_set(ret_tilt_deg=2.0)

    def kpis(cells: dict[str, object], step: int | None) -> dict[str, float]:
        tilt = cells["gNB_5G"]["ret_tilt_deg"]  # type: ignore[index]
        if tilt == 1.0:
            return cases._default_kpis(cell_goodput_mbps=100.0)
        return cases._default_kpis(interval_energy_j=-500.0 if (step or 0) >= switch_step else 100.0)

    plan = cases._build(
        objectives=["throughput_maximization", "energy_saving"],
        candidates_by_objective={"throughput_maximization": [throughput], "energy_saving": [energy]},
        predict_batch=cases._make_predict_batch(kpis),
    )
    assert plan.transition_time_steps == [switch_step]
