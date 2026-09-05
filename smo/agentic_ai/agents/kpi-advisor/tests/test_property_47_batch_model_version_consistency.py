"""Property 47: all probe batches must report one model version."""

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

import test_agent_baseline_and_probe_plan as cases  # noqa: E402
from smo.aimlfw.common.models import CellParameters, ParameterSet  # noqa: E402

import agent  # noqa: E402


# **Property 47: 배치 간 모델 버전 일치 검증**
# **Validates: Requirement 7.12**
@settings(max_examples=100, deadline=None)
@given(first_version=st.integers(min_value=1, max_value=1000), delta=st.integers(min_value=1, max_value=1000))
def test_mixed_batch_versions_are_rejected(first_version: int, delta: int) -> None:
    baseline = ParameterSet(
        cells={f"cell_{index}": CellParameters.model_validate(cases._cell_parameters()) for index in range(13)}
    )
    plan = agent.build_probe_plan(baseline, "baseline-property-47")
    calls = 0

    def predict(*, parameter_sets: list[dict[str, Any]], time_step: int | None = None) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        values = [{"cell_goodput_mbps": 10.0} for _ in parameter_sets]
        version = first_version if calls == 1 else first_version + delta
        return cases._batch_response(model_version=version, target_kpi_values=values)

    with pytest.raises(agent.ProbeExecutionError) as excinfo:
        agent.execute_probe_plan(plan, predict_batch=predict)
    assert excinfo.value.error_code == "model_version_mismatch"
    assert excinfo.value.details["expected_model_version"] == first_version
