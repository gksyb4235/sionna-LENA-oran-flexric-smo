"""Property 58: fewer than two usable parameters yields an empty conflict list."""

import sys
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

AGENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(REPO_ROOT), str(AGENT_DIR)]

from smo.aimlfw.common.models import MarginalEffectRecord  # noqa: E402

from conflict_analyzer import analyze  # noqa: E402


# **Property 58: 유효 파라미터 부족 시 빈 목록**
# **Validates: Requirement 9.10**
@settings(max_examples=100, deadline=None)
@given(valid_count=st.integers(min_value=0, max_value=1))
def test_fewer_than_two_valid_effects_reports_reason(valid_count: int) -> None:
    effects = [
        MarginalEffectRecord(
            cell_id="cell",
            control_parameter="tx_power_dbm",
            target_kpi="kpi",
            value_percent=8.0 if valid_count else None,
        )
    ]
    result = analyze(effects, 5.0)
    assert result.indirect_conflicts == []
    assert result.insufficient_parameters[0].valid_control_parameter_count == valid_count
    assert result.insufficient_parameters[0].reason
