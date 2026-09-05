"""Property 60: unspecified objectives select the fewest threshold violations."""

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


# **Property 60: objective 미지정 시 위반 최소 후보 선택**
# **Validates: Requirement 10.2**
@settings(max_examples=100, deadline=None)
@given(reverse=st.booleans())
def test_unspecified_mode_selects_the_zero_violation_candidate(reverse: bool) -> None:
    good = cases._parameter_set(tx_power_dbm=30.0)
    bad = cases._parameter_set(tx_power_dbm=46.0)
    candidates = [bad, good] if reverse else [good, bad]

    def kpis(cells: dict[str, object], _step: int | None) -> dict[str, float]:
        power = cells["gNB_5G"]["tx_power_dbm"]  # type: ignore[index]
        return cases._default_kpis(cell_goodput_mbps=-1.0) if power == 46.0 else cases._default_kpis()

    plan = cases._build(
        objectives=[],
        candidates_by_objective={"pool": candidates},
        predict_batch=cases._make_predict_batch(kpis),
    )
    assert all(item.parameter_set == good for item in plan.assignments)
    assert all(item.xapp_objective == "unspecified" for item in plan.assignments)
