"""Property 79: Korean intents require Korean rationale text."""

import pytest
from _policy_test_support import fixed_parameter_set  # noqa: F401 - initializes repository paths.
from hypothesis import given, settings
from hypothesis import strategies as st

from schemas import InvokeResponse, validate_response_for_intent


# **Property 79: 한국어 응답 텍스트 제약**
# **Validates: Requirement 12.6**
@settings(max_examples=100, deadline=None)
@given(suffix=st.text(alphabet=st.characters(whitelist_categories=("Ll",)), max_size=20))
def test_korean_intent_rejects_non_korean_rationale(suffix: str) -> None:
    response = InvokeResponse(
        degradation_verdict="unknown",
        recommendations=[],
        evidence_record_id="evidence-1",
        rationale_summary=f"No evidence {suffix}".strip(),
    )
    with pytest.raises(ValueError, match="Korean"):
        validate_response_for_intent(response, "KPI 영향을 분석해 주세요")

    korean = response.model_copy(update={"rationale_summary": f"판단 근거입니다 {suffix}"})
    validate_response_for_intent(korean, "KPI 영향을 분석해 주세요")
