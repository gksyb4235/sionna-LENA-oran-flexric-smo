"""Property 71: A1 HTTP 4xx/5xx responses are never retried."""

import pytest
from _policy_test_support import MockA1, install_mock, uninstall_mock
from hypothesis import given, settings
from hypothesis import strategies as st

import policy_manager


# **Property 71: A1 4xx/5xx 무재시도 실패**
# **Validates: Requirement 11.9**
@settings(max_examples=100, deadline=None)
@given(status_code=st.integers(min_value=400, max_value=599))
def test_registration_http_error_is_attempted_once(status_code: int) -> None:
    mock = MockA1(status_code=status_code)
    install_mock(mock)
    try:
        with pytest.raises(policy_manager.PolicyManagerError) as excinfo:
            policy_manager.register_policy_type()
        assert excinfo.value.error_code == "a1_http_error"
        assert mock.requests.count(("PUT", "/a1-p/policytypes/20100")) == 1
        attempts = policy_manager.get_publish_attempts()
        assert len(attempts) == 1 and attempts[0].status_code == status_code
    finally:
        uninstall_mock()
