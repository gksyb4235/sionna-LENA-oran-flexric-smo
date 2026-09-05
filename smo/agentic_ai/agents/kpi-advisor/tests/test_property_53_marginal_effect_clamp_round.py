"""Property 53: marginal effects are clamped and rounded to one decimal."""

import sys
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

AGENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(REPO_ROOT), str(AGENT_DIR)]

from smo.aimlfw.common.models import MarginalEffectRecord  # noqa: E402

from conflict_analyzer import analyze  # noqa: E402


# **Property 53: Marginal_Effect 클램프와 반올림**
# **Validates: Requirement 9.1**
@settings(max_examples=100, deadline=None)
@given(
    positive=st.floats(min_value=5.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    negative=st.floats(min_value=-1000.0, max_value=-5.0, allow_nan=False, allow_infinity=False),
)
def test_effects_are_clamped_and_rounded(positive: float, negative: float) -> None:
    effects = [
        MarginalEffectRecord.model_construct(
            cell_id="cell", control_parameter="tx_power_dbm", target_kpi="kpi", value_percent=positive
        ),
        MarginalEffectRecord.model_construct(
            cell_id="cell", control_parameter="ret_tilt_deg", target_kpi="kpi", value_percent=negative
        ),
    ]
    conflict = analyze(effects, 5.0).indirect_conflicts[0]
    assert conflict.marginal_effect_a == round(max(-100.0, min(100.0, negative)), 1)
    assert conflict.marginal_effect_b == round(max(-100.0, min(100.0, positive)), 1)
