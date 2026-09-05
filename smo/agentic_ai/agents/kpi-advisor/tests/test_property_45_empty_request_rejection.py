"""Property 45: empty probe requests are rejected before prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

AGENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(REPO_ROOT), str(AGENT_DIR)]

import agent  # noqa: E402
from schemas import XAppParameterRequest  # noqa: E402


class _EmptyStore:
    def records(self, *_args: object, **_kwargs: object) -> list[object]:
        return []


# **Property 45: 빈 요청 거부**
# **Validates: Requirement 7.8**
@settings(max_examples=100, deadline=None)
@given(empty_cells=st.booleans())
def test_empty_request_is_rejected_before_inference(empty_cells: bool) -> None:
    request = XAppParameterRequest.model_construct(
        target_cells=[] if empty_cells else ["gNB_5G"],
        parameter_overrides={},
        xapp_objective=None,
    )
    with pytest.raises(agent.ProbeExecutionError) as excinfo:
        agent.build_and_execute_probe_plan(request, feature_store=_EmptyStore())
    assert excinfo.value.error_code == "empty_request"
