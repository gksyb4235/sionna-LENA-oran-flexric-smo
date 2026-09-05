"""Property 80: public responses contain at most ten recommendations."""

import pytest
from _policy_test_support import fixed_parameter_set
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from schemas import InvokeResponse, Recommendation


# **Property 80: 추천 목록 개수 제약**
# **Validates: Requirement 12.7**
@settings(max_examples=100, deadline=None)
@given(count=st.integers(min_value=0, max_value=15))
def test_recommendation_count_contract(count: int) -> None:
    recommendations = [
        Recommendation(variant_id=f"variant-{index}", parameter_set=fixed_parameter_set()) for index in range(count)
    ]
    kwargs = {
        "degradation_verdict": "acceptable",
        "recommendations": recommendations,
        "evidence_record_id": "evidence-1",
        "rationale_summary": "valid rationale",
    }
    if count <= 10:
        assert len(InvokeResponse(**kwargs).recommendations) == count
    else:
        with pytest.raises(ValidationError):
            InvokeResponse(**kwargs)
