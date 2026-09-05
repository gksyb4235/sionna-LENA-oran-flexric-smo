"""Property 93: assembled Evidence Records contain every available decision input."""

from _evidence_test_support import record
from hypothesis import given, settings
from hypothesis import strategies as st


# **Property 93: Evidence_Record 필드 완전성**
# **Validates: Requirements 15.1, 15.2, 15.3**
@settings(max_examples=100, deadline=None)
@given(with_temporal=st.booleans())
def test_evidence_contains_complete_model_probe_prediction_and_decision_context(with_temporal: bool) -> None:
    evidence = record(with_temporal=with_temporal)
    assert evidence.evidence_id.startswith("evidence-")
    assert evidence.model_name == "gnn" and evidence.model_version == 1
    assert evidence.probe_plan.baseline and evidence.predictions
    assert len(evidence.applied_thresholds) == 10
    assert evidence.judgment_details
    assert evidence.marginal_effects and evidence.indirect_conflicts
    assert (evidence.temporal_plan is not None) is with_temporal
