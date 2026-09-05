"""Property 97: response recommendation KPI numbers are projected from stored evidence."""

from _evidence_test_support import Collection, record
from hypothesis import given, settings
from hypothesis import strategies as st

from evidence_store import EvidenceRepository
from schemas import InvokeResponse, Recommendation


# **Property 97: 응답 KPI 필드는 Evidence_Record의 부분집합**
# **Validates: Requirement 15.7**
@settings(max_examples=100, deadline=None)
@given(_case=st.none())
def test_response_recommendations_are_exactly_stored_values(_case: None) -> None:
    collection = Collection()
    repository = EvidenceRepository(collection)
    stored = repository.save(record(with_temporal=False))
    loaded = repository.get(stored.evidence_id)
    assert loaded is not None
    response = InvokeResponse(
        degradation_verdict=loaded.degradation_verdict,
        recommendations=[Recommendation.model_validate(item) for item in loaded.recommendations],
        evidence_record_id=loaded.evidence_id,
        rationale_summary="Stored evidence projection",
    )
    assert response.evidence_record_id == loaded.evidence_id
    assert [item.model_dump(mode="python") for item in response.recommendations] == loaded.recommendations
