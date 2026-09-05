"""Property 76: A1 connection failures stop after exactly three attempts."""

import sys
from pathlib import Path

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

AGENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(REPO_ROOT), str(AGENT_DIR)]

import policy_manager  # noqa: E402


# **Property 76: 연결 실패 재시도 한도**
# **Validates: Requirement 11.15**
@settings(max_examples=100, deadline=None)
@given(_case=st.none())
def test_connection_failure_is_attempted_exactly_three_times(_case: None) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("unreachable", request=request)

    policy_manager.set_client_factory(
        lambda: httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a1")
    )
    policy_manager.clear_publish_attempts()
    try:
        with pytest.raises(policy_manager.PolicyManagerError) as excinfo:
            policy_manager._request("PUT", "/a1-p/policytypes/{policy_type_id}", {"policy_type_id": 20100})
        assert excinfo.value.error_code == "a1_unreachable"
        assert calls == 3
        assert len(policy_manager.get_publish_attempts()) == 3
    finally:
        policy_manager.reset_client_factory()
        policy_manager.clear_publish_attempts()
