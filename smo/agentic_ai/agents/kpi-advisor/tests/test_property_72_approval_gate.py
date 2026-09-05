"""Property 72: degrading/unknown policies require explicit approval before any A1 call."""

from _policy_test_support import MockA1, fixed_parameter_set, install_mock, uninstall_mock
from hypothesis import given, settings
from hypothesis import strategies as st

import policy_manager


# **Property 72: 승인 게이팅**
# **Validates: Requirement 11.10**
@settings(max_examples=100, deadline=None)
@given(verdict=st.sampled_from(["degrading", "unknown"]))
def test_unapproved_risky_verdict_makes_no_a1_request(verdict: str) -> None:
    mock = MockA1()
    install_mock(mock)
    try:
        result = policy_manager.publish_parameter_set(
            fixed_parameter_set(),
            list(policy_manager.FIXED_CELLS),
            degradation_verdict=verdict,
            violated_threshold_kpis=["cell_goodput_mbps"],
        )
        assert isinstance(result, policy_manager.PendingApproval)
        assert result.status == "pending_approval"
        assert mock.requests == []
    finally:
        uninstall_mock()
