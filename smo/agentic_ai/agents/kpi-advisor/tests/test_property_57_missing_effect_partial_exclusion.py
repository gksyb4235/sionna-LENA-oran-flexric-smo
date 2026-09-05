"""Property 57: missing effects exclude only pairs containing that parameter."""

import sys
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

AGENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(REPO_ROOT), str(AGENT_DIR)]

from smo.aimlfw.common.models import MarginalEffectRecord  # noqa: E402

from conflict_analyzer import analyze  # noqa: E402


# **Property 57: 누락 파라미터의 부분 배제**
# **Validates: Requirement 9.8**
@settings(max_examples=100, deadline=None)
@given(missing=st.sampled_from(["tx_power_dbm", "ttt_ms", "hysteresis_db"]))
def test_unavailable_parameter_does_not_hide_other_conflicts(missing: str) -> None:
    effects = [
        MarginalEffectRecord(cell_id="cell", control_parameter=missing, target_kpi="kpi", value_percent=None),
        MarginalEffectRecord(cell_id="cell", control_parameter="ret_tilt_deg", target_kpi="kpi", value_percent=8.0),
        MarginalEffectRecord(cell_id="cell", control_parameter="cio_bias_db", target_kpi="kpi", value_percent=-8.0),
    ]
    result = analyze(effects, 5.0)
    assert ("cell", missing, "kpi") in result.unavailable_control_parameters
    assert len(result.indirect_conflicts) == 1
    assert missing not in {
        result.indirect_conflicts[0].control_parameter_a,
        result.indirect_conflicts[0].control_parameter_b,
    }
