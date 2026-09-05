"""Property 67: one prediction failure aborts the whole temporal plan."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

TEST_DIR = Path(__file__).resolve().parent
AGENT_DIR = TEST_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(REPO_ROOT), str(AGENT_DIR), str(TEST_DIR)]

import test_temporal_scheduler_smoke as cases  # noqa: E402

import temporal_scheduler  # noqa: E402


# **Property 67: 예측 불가 시 무부분계획 오류**
# **Validates: Requirement 10.13**
@settings(max_examples=100, deadline=None)
@given(failing_step=st.integers(min_value=0, max_value=4))
def test_any_failed_step_raises_instead_of_returning_partial_plan(failing_step: int) -> None:
    success = cases._make_predict_batch(lambda _cells, _step: cases._default_kpis())

    def predict(parameter_sets: list[dict[str, Any]], time_step: int | None = None) -> dict[str, Any]:
        if time_step == failing_step:
            return {"error_code": "inference_unavailable", "message": "boom", "details": {}}
        return success(parameter_sets, time_step)

    with pytest.raises(temporal_scheduler.TemporalSchedulerError) as excinfo:
        cases._build(
            objectives=["throughput_maximization"],
            candidates_by_objective={"throughput_maximization": [cases._parameter_set()]},
            predict_batch=predict,
        )
    assert excinfo.value.error_code == "prediction_unavailable"
