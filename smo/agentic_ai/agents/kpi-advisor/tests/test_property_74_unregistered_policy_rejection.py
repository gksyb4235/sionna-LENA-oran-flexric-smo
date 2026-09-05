"""Property 74: publishing against an unregistered Policy Type creates no instance."""

import pytest
from _policy_test_support import MockA1, fixed_parameter_set, install_mock, uninstall_mock
from hypothesis import given, settings
from hypothesis import strategies as st

import policy_manager


# **Property 74: 미등록 Policy_Type 발행 거부**
# **Validates: Requirement 11.13**
@settings(max_examples=100, deadline=None)
@given(policy_type_id=st.integers(min_value=1, max_value=2_000_000_000))
def test_unregistered_type_is_rejected_before_instance_put(policy_type_id: int) -> None:
    mock = MockA1()
    install_mock(mock)
    try:
        with pytest.raises(policy_manager.PolicyManagerError) as excinfo:
            policy_manager.publish_parameter_set(
                fixed_parameter_set(), list(policy_manager.FIXED_CELLS), policy_type_id=policy_type_id
            )
        assert excinfo.value.error_code == "policy_type_not_registered"
        assert not any(method == "PUT" and "/policies/" in path for method, path in mock.requests)
    finally:
        uninstall_mock()
