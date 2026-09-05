"""Property 81: shell and kubectl command strings are rejected from public responses."""

import pytest
from _policy_test_support import fixed_parameter_set  # noqa: F401 - initializes repository paths.
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from schemas import InvokeResponse


# **Property 81: 금지 문자열 부재**
# **Validates: Requirement 12.8**
@settings(max_examples=100, deadline=None)
@given(command=st.sampled_from(["kubectl get pods", "$ rm -rf /tmp/x", "docker ps", "curl http://x", "a && b"]))
def test_command_like_rationale_is_rejected(command: str) -> None:
    with pytest.raises(ValidationError):
        InvokeResponse(
            degradation_verdict="unknown",
            recommendations=[],
            evidence_record_id="evidence-1",
            rationale_summary=command,
        )
