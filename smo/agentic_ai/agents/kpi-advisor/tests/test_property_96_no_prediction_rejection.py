"""Property 96: a judgment cannot be recorded without prediction evidence."""

import dataclasses

import pytest
from _evidence_test_support import components
from hypothesis import given, settings
from hypothesis import strategies as st

import agent
import evidence_store


# **Property 96: 예측값 없는 판정 거부**
# **Validates: Requirement 15.6**
@settings(max_examples=100, deadline=None)
@given(_case=st.none())
def test_empty_baseline_prediction_rejects_evidence_assembly(_case: None) -> None:
    values = components(with_temporal=False)
    probe_result = values["probe_result"]
    empty_prediction = agent.ProbePredictionResult(target_kpi={}, cell_kpi={})
    values["probe_result"] = dataclasses.replace(probe_result, baseline_prediction=empty_prediction)
    with pytest.raises(evidence_store.EvidenceStoreError) as excinfo:
        evidence_store.assemble_evidence_record(**values)
    assert excinfo.value.error_code == "no_prediction_evidence"
