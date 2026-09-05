"""Property 64: an empty candidate input is derived from usable Feature Store records."""

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


# **Property 64: 후보 0개 시 Feature_Store 기반 도출**
# **Validates: Requirements 10.8, 10.9**
@settings(max_examples=100, deadline=None)
@given(power=st.integers(min_value=30, max_value=46))
def test_feature_store_fallback_preserves_step_parameter_sets(power: int) -> None:
    records = [
        cases._feature_record(cell_id="gNB_5G", time_step=step).model_copy(
            update={"control_parameters": cases._cell_parameters(tx_power_dbm=float(power))}
        )
        for step in range(5)
    ]
    plan = cases._build(
        objectives=[],
        candidates_by_objective={},
        feature_store=cases._FakeFeatureStore(records),
        predict_batch=cases._make_predict_batch(lambda _cells, _step: cases._default_kpis()),
    )
    assert len(plan.assignments) == 5
    assert all(item.parameter_set.cells["gNB_5G"].tx_power_dbm == power for item in plan.assignments)
