"""Property 48: KPI bounds are inclusive and violations are strict outside them."""

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


def _prediction(values: dict[str, float]) -> agent.ProbePredictionResult:
    return agent.ProbePredictionResult(
        target_kpi=values,
        cell_kpi={"gNB_5G": values},
        confidence_interval_width={name: 0.0 for name in values},
    )


# **Property 48: Degradation_Verdict 경계 포함 판정**
# **Validates: Requirements 8.2, 8.3**
@settings(max_examples=100, deadline=None)
@given(target_kpi=st.sampled_from(TARGET_KPIS), use_lower=st.booleans())
def test_threshold_boundary_is_acceptable_and_outside_is_degrading(target_kpi: str, use_lower: bool) -> None:
    thresholds = cases._default_kpi_threshold_config()
    threshold = thresholds.thresholds[target_kpi]
    bound = threshold.lower_bound if use_lower else threshold.upper_bound
    if bound is None:
        bound = threshold.upper_bound if use_lower else threshold.lower_bound
        use_lower = not use_lower
    assert bound is not None

    at_boundary = cases._all_acceptable_target_kpi(**{target_kpi: bound})
    assert agent.judge_degradation_verdict(_prediction(at_boundary), thresholds).verdict == "acceptable"

    outside = bound - 0.1 if use_lower else bound + 0.1
    violated = cases._all_acceptable_target_kpi(**{target_kpi: outside})
    judgment = agent.judge_degradation_verdict(_prediction(violated), thresholds)
    assert judgment.verdict == "degrading"
    assert target_kpi in {item.target_kpi for item in judgment.violations}
