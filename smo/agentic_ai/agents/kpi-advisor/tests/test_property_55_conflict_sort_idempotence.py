"""Property 55: conflict output is deterministic for every input permutation."""

import sys
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

AGENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(REPO_ROOT), str(AGENT_DIR)]

from smo.aimlfw.common.models import MarginalEffectRecord  # noqa: E402

from conflict_analyzer import analyze  # noqa: E402

EFFECTS = [
    MarginalEffectRecord(cell_id="b", control_parameter="tx_power_dbm", target_kpi="z", value_percent=8.0),
    MarginalEffectRecord(cell_id="b", control_parameter="ret_tilt_deg", target_kpi="z", value_percent=-8.0),
    MarginalEffectRecord(cell_id="a", control_parameter="hysteresis_db", target_kpi="a", value_percent=9.0),
    MarginalEffectRecord(cell_id="a", control_parameter="cio_bias_db", target_kpi="a", value_percent=-9.0),
]


# **Property 55: 결정론적 정렬 재현**
# **Validates: Requirement 9.7**
@settings(max_examples=100, deadline=None)
@given(permuted=st.permutations(EFFECTS))
def test_permutations_produce_identical_sorted_conflicts(permuted: tuple[MarginalEffectRecord, ...]) -> None:
    expected = analyze(EFFECTS, 5.0)
    assert analyze(list(permuted), 5.0).indirect_conflicts == expected.indirect_conflicts
