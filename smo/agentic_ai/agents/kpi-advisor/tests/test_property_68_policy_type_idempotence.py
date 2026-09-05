"""Property 68: Policy Type registration is idempotent."""

from _policy_test_support import MockA1, install_mock, uninstall_mock
from hypothesis import given, settings
from hypothesis import strategies as st

import policy_manager


# **Property 68: Policy_Type 등록 멱등성**
# **Validates: Requirement 11.2**
@settings(max_examples=100, deadline=None)
@given(repeats=st.integers(min_value=1, max_value=5))
def test_registration_uses_at_most_one_put(repeats: int) -> None:
    mock = MockA1()
    install_mock(mock)
    try:
        assert [policy_manager.register_policy_type() for _ in range(repeats)] == [20100] * repeats
        puts = [item for item in mock.requests if item == ("PUT", "/a1-p/policytypes/20100")]
        assert len(puts) == 1
    finally:
        uninstall_mock()
