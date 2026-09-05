"""Property 69: one ParameterSet creates one instance and a TemporalPlan creates five."""

from _policy_test_support import MockA1, fixed_parameter_set, install_mock, temporal_plan, uninstall_mock
from hypothesis import given, settings
from hypothesis import strategies as st

import policy_manager


# **Property 69: Policy_Instance 개수 불변식**
# **Validates: Requirements 11.3, 11.4**
@settings(max_examples=100, deadline=None)
@given(temporal=st.booleans())
def test_publish_instance_count_matches_input_shape(temporal: bool) -> None:
    mock = MockA1()
    install_mock(mock)
    try:
        policy_manager.register_policy_type()
        if temporal:
            result = policy_manager.publish_temporal_plan(temporal_plan())
            assert isinstance(result, list) and len(result) == 5
        else:
            result = policy_manager.publish_parameter_set(fixed_parameter_set(), list(policy_manager.FIXED_CELLS))
            assert isinstance(result, policy_manager.PolicyInstanceResult)
        assert len(mock.instances[20100]) == (5 if temporal else 1)
    finally:
        uninstall_mock()
