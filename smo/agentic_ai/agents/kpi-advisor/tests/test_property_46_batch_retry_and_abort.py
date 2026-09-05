"""Property 46: each failed batch is retried within budget or aborts atomically."""

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

import agent  # noqa: E402


def _success(parameter_sets: list[dict[str, Any]]) -> dict[str, Any]:
    values = [{"cell_goodput_mbps": 10.0} for _ in parameter_sets]
    return cases._batch_response(target_kpi_values=values)


# **Property 46: 배치 재전송과 실패 중단**
# **Validates: Requirement 7.11**
@settings(max_examples=100, deadline=None)
@given(failures=st.integers(min_value=0, max_value=2))
def test_batch_succeeds_when_failure_count_is_within_retry_budget(failures: int) -> None:
    calls = 0

    def predict(*, parameter_sets: list[dict[str, Any]], time_step: int | None = None) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls <= failures:
            return cases._error_body()
        return _success(parameter_sets)

    result = agent.execute_probe_plan(cases._simple_plan(), predict_batch=predict, max_batch_retries=2)
    assert calls == failures + 1
    assert result.model_version == 1


@settings(max_examples=100, deadline=None)
@given(retries=st.integers(min_value=0, max_value=4))
def test_retry_exhaustion_returns_no_partial_probe_result(retries: int) -> None:
    calls = 0

    def predict(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return cases._error_body()

    with pytest.raises(agent.ProbeExecutionError) as excinfo:
        agent.execute_probe_plan(cases._simple_plan(), predict_batch=predict, max_batch_retries=retries)
    assert calls == retries + 1
    assert excinfo.value.error_code == "probe_batch_failed"
    assert excinfo.value.details["attempts"] == retries + 1
