"""Property 49: missing or unsupported confidence evidence yields unknown."""

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


# **Property 49: unknown 판정 조건**
# **Validates: Requirements 8.4, 8.12**
@settings(max_examples=100, deadline=None)
@given(target_kpi=st.sampled_from(TARGET_KPIS), confidence_absent=st.booleans())
def test_missing_or_wide_confidence_without_actual_evidence_is_unknown(
    target_kpi: str, confidence_absent: bool
) -> None:
    values = cases._all_acceptable_target_kpi()
    confidence = None
    if not confidence_absent:
        confidence = {name: 0.0 for name in TARGET_KPIS}
        value = values[target_kpi]
        if value == 0.0:
            value = 1.0
            values[target_kpi] = value
        confidence[target_kpi] = max(abs(value) * 0.6, 1.0)
    prediction = agent.ProbePredictionResult(
        target_kpi=values,
        cell_kpi={"gNB_5G": values},
        confidence_interval_width=confidence,
    )
    judgment = agent.judge_degradation_verdict(prediction, cases._default_kpi_threshold_config())
    assert judgment.verdict == "unknown"
    if not confidence_absent:
        assert target_kpi in {item.target_kpi for item in judgment.unknown_reasons}
