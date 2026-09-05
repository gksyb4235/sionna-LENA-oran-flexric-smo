"""Property 52: only variants with missing predictions are excluded."""

from __future__ import annotations

import sys
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

TEST_DIR = Path(__file__).resolve().parent
AGENT_DIR = TEST_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(REPO_ROOT), str(AGENT_DIR), str(TEST_DIR)]

import test_agent_threshold_judgment_and_recommendations as cases  # noqa: E402
from smo.aimlfw.common.constants import TARGET_KPIS  # noqa: E402

import agent  # noqa: E402


# **Property 52: 누락 예측값의 부분 배제**
# **Validates: Requirement 8.11**
@settings(max_examples=100, deadline=None)
@given(missing_kpi=st.sampled_from(TARGET_KPIS))
def test_missing_prediction_excludes_only_affected_variant(missing_kpi: str) -> None:
    baseline = cases._simple_baseline()
    missing_variant = cases._variant(
        baseline, cell_id="gNB_5G", parameter_name="tx_power_dbm", direction="+1", variant_id="missing"
    )
    good_variant = missing_variant.model_copy(update={"variant_id": "good"})
    missing_values = cases._all_acceptable_target_kpi()
    del missing_values[missing_kpi]
    missing_result = agent.ProbeVariantResult(
        variant=missing_variant,
        prediction=agent.ProbePredictionResult(target_kpi=missing_values, cell_kpi={"gNB_5G": missing_values}),
        percent_change={"cell_goodput_mbps": 10.0},
    )
    good_result = cases._variant_result(good_variant, cell_goodput_mbps=100.0, improvement_percent=10.0)
    outcome = cases._recommend(
        [missing_result, good_result],
        cases._default_kpi_threshold_config(),
        cases._default_objective_primary_kpi_config(),
    )
    assert [item.variant_id for item in outcome.recommendations] == ["good"]
    assert [(item.variant_id, item.reason) for item in outcome.excluded] == [("missing", "missing_prediction")]
