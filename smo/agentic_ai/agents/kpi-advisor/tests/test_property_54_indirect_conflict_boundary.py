"""Property 54: opposite effects at both inclusive thresholds conflict."""

import sys
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

AGENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(REPO_ROOT), str(AGENT_DIR)]

from smo.aimlfw.common.models import MarginalEffectRecord  # noqa: E402

from conflict_analyzer import analyze  # noqa: E402


# **Property 54: Indirect_Conflict 경계 포함 판정**
# **Validates: Requirements 9.2~9.4**
@settings(max_examples=100, deadline=None)
@given(threshold=st.integers(min_value=1, max_value=100).map(float))
def test_exact_opposite_thresholds_are_conflicts(threshold: float) -> None:
    effects = [
        MarginalEffectRecord(
            cell_id="cell", control_parameter="tx_power_dbm", target_kpi="kpi", value_percent=threshold
        ),
        MarginalEffectRecord(
            cell_id="cell", control_parameter="ret_tilt_deg", target_kpi="kpi", value_percent=-threshold
        ),
    ]
    assert len(analyze(effects, threshold).indirect_conflicts) == 1
    below = effects[1].model_copy(update={"value_percent": -(threshold - 0.1)})
    assert analyze([effects[0], below], threshold).indirect_conflicts == []
