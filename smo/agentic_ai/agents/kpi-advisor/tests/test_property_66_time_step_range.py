"""Property 66: public advisor requests reject Time_Step outside 0..4."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

TEST_DIR = Path(__file__).resolve().parent
AGENT_DIR = TEST_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(REPO_ROOT), str(AGENT_DIR), str(TEST_DIR)]

import test_temporal_scheduler_smoke as cases  # noqa: E402

from schemas import InvokeRequest  # noqa: E402


# **Property 66: Time_Step 범위 검증**
# **Validates: Requirement 10.12**
@settings(max_examples=100, deadline=None)
@given(step=st.one_of(st.integers(max_value=-1), st.integers(min_value=5, max_value=10000)))
def test_public_request_rejects_out_of_range_time_step(step: int) -> None:
    with pytest.raises(ValidationError):
        InvokeRequest(intent="evaluate", baseline_parameter_set=cases._parameter_set(), time_step=step)
