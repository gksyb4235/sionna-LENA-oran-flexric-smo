"""Property 65: equal scores resolve by objective name regardless of input order."""

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


# **Property 65: 동률 배정의 결정론적 해소**
# **Validates: Requirement 10.11**
@settings(max_examples=100, deadline=None)
@given(reverse=st.booleans())
def test_equal_objective_scores_choose_lexicographically_first_objective(reverse: bool) -> None:
    objectives = ["throughput_maximization", "energy_saving"]
    if reverse:
        objectives.reverse()
    plan = cases._build(
        objectives=objectives,
        candidates_by_objective={name: [cases._parameter_set()] for name in objectives},
        predict_batch=cases._make_predict_batch(
            lambda _cells, _step: cases._default_kpis(cell_goodput_mbps=10.0, interval_energy_j=-10.0)
        ),
    )
    assert all(item.xapp_objective == "energy_saving" for item in plan.assignments)
