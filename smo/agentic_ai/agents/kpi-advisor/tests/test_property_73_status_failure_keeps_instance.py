"""Property 73: a status failure never rolls back the created policy instance."""

from _policy_test_support import MockA1, fixed_parameter_set, install_mock, uninstall_mock
from hypothesis import given, settings
from hypothesis import strategies as st

import policy_manager


# **Property 73: 상태 조회 실패 시 인스턴스 유지**
# **Validates: Requirements 11.11, 11.12**
@settings(max_examples=100, deadline=None)
@given(_case=st.none())
def test_status_unavailable_preserves_created_instance(_case: None) -> None:
    mock = MockA1(status_available=False)
    install_mock(mock)
    try:
        policy_manager.register_policy_type()
        result = policy_manager.publish_parameter_set(fixed_parameter_set(), list(policy_manager.FIXED_CELLS))
        assert isinstance(result, policy_manager.PolicyInstanceResult)
        assert result.status == "status_unavailable"
        assert result.policy_instance_id in mock.instances[20100]
    finally:
        uninstall_mock()
