"""Property 94: every assembled Evidence Record receives a unique identifier."""

from _evidence_test_support import record
from hypothesis import given, settings
from hypothesis import strategies as st


# **Property 94: Evidence_Record 식별자 고유성**
# **Validates: Requirement 15.4**
@settings(max_examples=100, deadline=None)
@given(count=st.integers(min_value=2, max_value=20))
def test_generated_evidence_ids_are_unique(count: int) -> None:
    identifiers = [record(with_temporal=False).evidence_id for _ in range(count)]
    assert len(set(identifiers)) == count
