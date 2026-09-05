"""Property 95: ambiguous storage failure removes any partially inserted document."""

import pytest
from _evidence_test_support import Collection, record
from hypothesis import given, settings
from hypothesis import strategies as st

import evidence_store


# **Property 95: 저장 실패 원자성**
# **Validates: Requirement 15.5**
@settings(max_examples=100, deadline=None)
@given(_case=st.none())
def test_failed_insert_leaves_no_partial_document(_case: None) -> None:
    collection = Collection(fail_after_insert=True)
    repository = evidence_store.EvidenceRepository(collection)
    evidence = record(with_temporal=False)
    with pytest.raises(evidence_store.EvidenceStoreError) as excinfo:
        repository.save(evidence)
    assert excinfo.value.error_code == "evidence_storage_failed"
    assert collection.documents == {}
